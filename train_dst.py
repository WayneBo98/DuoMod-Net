import tempfile
print(tempfile.gettempdir())
import os
import sys
import logging
import random
from tqdm import tqdm
import argparse
from math import ceil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

# --- DST Adaptation: 假设 VNet_Decoupled 模型存在于 networks 文件夹中 ---
# from networks.vnet import VNet
from networks.vnet_dst import VNet_Decoupled # 替换为解耦模型
# --- End DST Adaptation ---

from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, sliding_window_inference
from utils.new_loss import DC_and_CE_loss, RobustCrossEntropyLoss
from utils.transforms import RandomCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg
from utils.config import Config

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='dst') # 实验名称改为 dst
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_2p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_2p')
parser.add_argument('-se', '--split_eval', type=str, default='valid')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=150)
# --- DST Adaptation: 添加新参数 ---
parser.add_argument('--confidence_thresh', type=float, default=0.95, help='Confidence threshold for pseudo-label masking')
parser.add_argument('--wce_w', type=float, default=1.0, help='Weight for the worst-case estimation loss')
# --- End DST Adaptation ---
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)

# --- 辅助函数区 ---

def dice_from_labels_gpu(pred_labels: torch.Tensor, gt_labels: torch.Tensor, num_classes: int):
    # ... (与 baseline 完全相同，为简洁省略)
    assert pred_labels.device == gt_labels.device
    pred = pred_labels.reshape(-1)
    gt   = gt_labels.reshape(-1)
    dices = []
    eps = 1e-8
    for c in range(1, num_classes):
        pred_c = (pred == c)
        gt_c   = (gt   == c)
        intersection = (pred_c & gt_c).sum()
        pred_sum = pred_c.sum()
        gt_sum = gt_c.sum()
        denominator = pred_sum + gt_sum
        if denominator == 0:
            dices.append(1.0)
        else:
            dice_val = (2.0 * intersection) / (denominator + eps)
            dices.append(dice_val.item())
    return dices


# --- DST Adaptation: 适配集成模型 ---
class ModelEnsemble(nn.Module):
    def __init__(self, model_A, model_B):
        super(ModelEnsemble, self).__init__()
        self.model_A = model_A
        self.model_B = model_B

    def forward(self, x):
        # 在验证时，只使用模型的主输出来进行集成
        out_A = self.model_A(x)
        out_B = self.model_B(x)
        return (out_A + out_B) / 2.0
# --- End DST Adaptation ---


def sigmoid_rampup(current, rampup_length):
    # ... (与 baseline 完全相同，为简洁省略)
    if rampup_length == 0: return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # ... (与 baseline 完全相同，为简洁省略)
    if args.cps_rampup:
        rampup_len = args.consistency_rampup if args.consistency_rampup is not None else args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, rampup_len)
    return args.cps_w

def kaiming_normal_init_weight(model):
    # ... (与 baseline 完全相同，为简洁省略)
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    # ... (与 baseline 完全相同，为简洁省略)
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def make_loss_function(name, weight=None):
    # ... (与 baseline 完全相同，为简洁省略)
    if name == 'ce': return RobustCrossEntropyLoss()
    elif name == 'wce': return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice': return DC_and_CE_loss(n_classes=config.num_cls)
    elif name == 'w_ce+dice': return DC_and_CE_loss(n_classes=config.num_cls, w_dc=weight, w_ce=weight)
    else: raise ValueError(name)

# --- DST Adaptation: 定义新的损失函数 ---
class MaskedCrossEntropyLoss(nn.Module):
    """带掩码的交叉熵损失，用于处理置信度筛选"""
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, pred, target, mask):
        # target 是 [B, D, H, W] 的 long a tensor
        # mask 是 [B, D, H, W] 的 float tensor
        if not (target.size() == mask.size()):
            raise ValueError(f"Target size {target.size()} must be the same as mask size {mask.size()}")
        
        loss = F.cross_entropy(pred, target, reduction='none')
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

def shift_log(x, offset=1e-6):
    """为保证数值稳定性，先平移再计算log"""
    return torch.log(torch.clamp(x + offset, max=1.))

class WorstCaseEstimationLoss(nn.Module):

    def __init__(self):
        super(WorstCaseEstimationLoss, self).__init__()

    def forward(self, y_l, y_l_adv, y_u, y_u_adv):

        prediction_l = torch.argmax(y_l.detach(), dim=1, keepdim=True).long()
        if len(prediction_l.shape) == len(y_l_adv.shape):
            assert prediction_l.shape[1] == 1
            prediction_l = prediction_l[:, 0]
        loss_l = F.cross_entropy(y_l_adv, prediction_l)

        prediction_u = torch.argmax(y_u.detach(), dim=1, keepdim=True).long()
        if len(prediction_u.shape) == len(y_u_adv.shape):
            assert prediction_u.shape[1] == 1
            prediction_u = prediction_u[:, 0]
        loss_u = F.nll_loss(shift_log(1. - F.softmax(y_u_adv, dim=1)), prediction_u)

        return 2 * loss_l + loss_u
# --- End DST Adaptation ---


def make_model_all():
    # --- DST Adaptation: 使用 VNet_Decoupled 模型 ---
    model = VNet_Decoupled(
        n_channels=config.num_channels, n_classes=config.num_cls, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=True
    ).cuda()
    # --- End DST Adaptation ---
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    return model, optimizer

if __name__ == '__main__':
    # --- 初始化和日志设置 (与 baseline 完全相同) ---
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    snapshot_path = f'./logs/{args.task}'+args.split_labeled[-2:]+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    # snapshot_path = f'./logs/{args.task}'+'10p'+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'), level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # === 1. 数据加载 (与 baseline 完全相同) ===
    train_transform = transforms.Compose([
        RandomCrop(config.patch_size), RandomFlip_LR(), RandomFlip_UD(), ToTensor()
    ])
    eval_transform = transforms.Compose([ToTensor()])
    db_labeled = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=train_transform)
    db_unlabeled = OrganSeg(task=args.task, split=args.split_unlabeled, unlabeled=True, num_cls=config.num_cls, transform=train_transform)
    db_eval = OrganSeg(task=args.task, split=args.split_eval, pre_load=True, num_cls=config.num_cls, transform=eval_transform)
    def worker_init_fn(worker_id):
        random.seed(SEED + worker_id)
    sampler = torch.utils.data.RandomSampler(db_labeled, replacement=True, num_samples=len(db_unlabeled))
    labeled_loader = DataLoader(
        db_labeled, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn
    )
    unlabeled_loader = DataLoader(
        db_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn
    )
    eval_loader = DataLoader(db_eval, batch_size=1, shuffle=False, pin_memory=True)
    logging.info(f'{len(unlabeled_loader)} iterations per epoch')

    # --- 模型、优化器和损失函数设置 ---
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)
    logging.info(optimizer_A)
    
    ensemble_model = ModelEnsemble(model_A, model_B)

    # --- DST Adaptation: 初始化新的损失函数 ---
    loss_func = make_loss_function(args.sup_loss)
    cps_loss_func = MaskedCrossEntropyLoss()
    worst_case_criterion = WorstCaseEstimationLoss()
    # --- End DST Adaptation ---
    amp_grad_scaler = GradScaler()

    # --- 主训练循环 ---
    best_eval = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    for epoch_num in range(args.max_epoch + 1):
        # --- DST Adaptation: 添加新的 loss 列表用于记录 ---
        loss_list, loss_sup_list, loss_cps_list, loss_wce_list = [], [], [], []
        # --- End DST Adaptation ---
        model_A.train()
        model_B.train()
        cps_w = get_current_consistency_weight(epoch_num)

        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader), total=len(unlabeled_loader), desc=f"epoch {epoch_num} training"):
            optimizer_A.zero_grad()
            optimizer_B.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)

            with autocast('cuda'):
                # --- DST Adaptation: 核心训练逻辑替换 ---

                # 1. 前向传播，获取三个输出
                output_A, output_A_pseudo, output_A_worst = model_A(image)
                output_B, output_B_pseudo, output_B_worst = model_B(image)
                del image

                # 2. 分割有标签和无标签的输出
                num_labeled = image_l.shape[0]
                output_A_l, output_A_u = output_A[:num_labeled], output_A[num_labeled:]
                output_B_l, output_B_u = output_B[:num_labeled], output_B[num_labeled:]
                
                # 3. 计算监督损失 (与CPS相同)
                loss_sup = loss_func(output_A_l, label_l) + loss_func(output_B_l, label_l)

                # 4. 计算带置信度掩码的CPS损失
                # 从伪标签头生成 target 和 confidence
                confidence_B, target_B = F.softmax(output_B_pseudo.detach(), dim=1).max(dim=1)
                confidence_A, target_A = F.softmax(output_A_pseudo.detach(), dim=1).max(dim=1)
                
                # 生成掩码
                mask_B = (confidence_B > args.confidence_thresh).float()
                mask_A = (confidence_A > args.confidence_thresh).float()

                loss_cps = cps_loss_func(output_A, target_B, mask_B) + cps_loss_func(output_B, target_A, mask_A)

                # 5. 计算最坏情况估计损失
                output_A_worst_l, output_A_worst_u = output_A_worst[:num_labeled], output_A_worst[num_labeled:]
                output_B_worst_l, output_B_worst_u = output_B_worst[:num_labeled], output_B_worst[num_labeled:]
                
                loss_wce = worst_case_criterion(output_A_l, output_A_worst_l, output_A_u, output_A_worst_u) + \
                           worst_case_criterion(output_B_l, output_B_worst_l, output_B_u, output_B_worst_u)

                # 6. 计算总损失
                loss = loss_sup + cps_w * (loss_cps + loss_wce)

                # --- End DST Adaptation ---

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer_A)
            amp_grad_scaler.step(optimizer_B)
            amp_grad_scaler.update()

            # --- DST Adaptation: 调用模型特有的 step 方法 (如果存在) ---
            # 这一步取决于 VNet_Decoupled 的内部实现，如果它需要更新内部状态，则需要调用
            if hasattr(model_A, 'step') and callable(getattr(model_A, 'step')):
                model_A.step()
            if hasattr(model_B, 'step') and callable(getattr(model_B, 'step')):
                model_B.step()
            # --- End DST Adaptation ---

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())
            # --- DST Adaptation: 记录新的 loss ---
            loss_wce_list.append(loss_wce.item())
            # --- End DST Adaptation ---
        
        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/total', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/supervised', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/consistency', np.mean(loss_cps_list), epoch_num)
        # --- DST Adaptation: 记录新的 loss 到 tensorboard ---
        writer.add_scalar('loss/worst_case', np.mean(loss_wce_list), epoch_num)
        # --- End DST Adaptation ---
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list):.4f}, cps_w: {cps_w:.4f}, lr: {get_lr(optimizer_A):.6f}')
        
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        
        # === 2. 验证循环 (使用新框架的高效验证逻辑) ===
        if (epoch_num > 0 and epoch_num <150 and epoch_num % 10 == 0) or (epoch_num >= 150 and epoch_num % 1 == 0):
            dice_list_for_epoch = [[] for _ in range(config.num_cls - 1)]
            model_A.eval()
            model_B.eval()

            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"epoch {epoch_num} validation"):
                    image, gt = fetch_data(batch)
                    
                    # 使用滑窗推理，ModelEnsemble 已被适配
                    pred_indices = sliding_window_inference(
                        image=image,
                        model=ensemble_model,
                        patch_size=config.patch_size,
                        num_classes=config.num_cls,
                        overlap=0.25
                    )
                    gt_indices = gt.squeeze(1).long()
                    
                    dice_per_class = dice_from_labels_gpu(
                        pred_labels=pred_indices,
                        gt_labels=gt_indices,
                        num_classes=config.num_cls
                    )
                    
                    for i in range(len(dice_per_class)):
                        dice_list_for_epoch[i].append(dice_per_class[i])
                        
                    del image, gt, pred_indices, gt_indices

            # --- 结果聚合与保存逻辑 (与 baseline 完全相同) ---
            dice_mean_per_class = [np.mean(d) for d in dice_list_for_epoch if d]
            if dice_mean_per_class:
                avg_dice = np.mean(dice_mean_per_class)
                logging.info(f'Validation epoch {epoch_num}, Avg Dice: {avg_dice:.4f}')
                
                # class_names= ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'pancreas', 'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate/uterus']
                class_names= ['liver', 'spleen', 'kidney L', 'kidney R', 'stomach', 'gallbladder', 'esophagus',  'pancreas', 'duodenum', 'colon', 'intestine', 'adrenal', 'rectum', 'bladder', 'femur']
                dice_dict = {name: f"{d:.4f}" for name, d in zip(class_names, dice_mean_per_class)}
                logging.info(f'Per-class Dice: {dice_dict}')

                if avg_dice > best_eval:
                    best_eval = avg_dice
                    best_epoch = epoch_num
                    epochs_no_improve = 0
                    save_path = os.path.join(snapshot_path, 'ckpts/best_model.pth')
                    torch.save({'A': model_A.state_dict(), 'B': model_B.state_dict()}, save_path)
                    logging.info(f'New best model saved to {save_path}')
                else:
                    epochs_no_improve += 1
                logging.info(f'\t Best eval dice is {best_eval:.4f} in epoch {best_epoch}')
                logging.info(f'\t Epochs with no improvement: {epochs_no_improve} / {config.early_stop_patience}')

                if epochs_no_improve >= config.early_stop_patience:
                    logging.info(f'--- Early stopping triggered after {epochs_no_improve} validation cycles with no improvement. ---')
                    break
    
    logging.info("--- Training Finished ---")
    writer.close()