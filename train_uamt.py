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
from torch.amp import GradScaler, autocast  # 统一使用 torch.amp（PyTorch 2.0+）

# --- Models & Utils (请确保路径正确) ---
from networks.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr,sliding_window_inference
from utils.new_loss import DC_and_CE_loss, RobustCrossEntropyLoss
from utils.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg  # 统一使用 OrganSeg，适配 task
from utils.config import Config

# --- 命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='uamt')
parser.add_argument('--exp_name', type=str, default='base')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_2p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_2p')
parser.add_argument('-se', '--split_eval', type=str, default='valid')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')  # 未直接使用，保留兼容
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', action='store_true', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=150)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)

# --- 辅助函数 ---
def dice_from_labels_gpu(pred_labels: torch.Tensor, gt_labels: torch.Tensor, num_classes: int):
    """
    一个内存高效的Dice计算函数，在GPU上运行 (修正版)。
    使用 .reshape() 代替 .view() 来处理可能的非连续张量。
    """
    assert pred_labels.device == gt_labels.device
    
    pred = pred_labels.reshape(-1) # 使用 .reshape() 替代 .view()
    gt   = gt_labels.reshape(-1)  # 使用 .reshape() 替代 .view()
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

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0: return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    if args.cps_rampup:
        rampup_len = args.consistency_rampup if args.consistency_rampup is not None else args.max_epoch
        return args.cps_w * sigmoid_rampup(epoch, rampup_len)
    return args.cps_w

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):  # 未使用，保留结构
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def make_loss_function(name, weight=None):
    if name == 'ce': return RobustCrossEntropyLoss()
    elif name == 'wce': return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice': return DC_and_CE_loss(n_classes=config.num_cls)
    elif name == 'w_ce+dice': return DC_and_CE_loss(n_classes=config.num_cls, w_dc=weight, w_ce=weight)
    else: raise ValueError(name)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=(1 - alpha))

def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss

# --- 模型构建 ---
def make_model_all(ema=False):
    model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=True
    ).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model, optimizer

if __name__ == '__main__':
    # --- 初始化 ---
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # snapshot_path = f'./logs/{args.task}'+'10p'+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    snapshot_path = f'./logs/{args.task}'+args.split_labeled[-2:]+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'), level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # === 1. 数据加载 (沿用CPS结构) ===
    train_transform = transforms.Compose([
        RandomCrop(config.patch_size), RandomFlip_LR(), RandomFlip_UD(), ToTensor()
    ])
    eval_transform = transforms.Compose([ToTensor()])

    db_labeled = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=train_transform)
    db_unlabeled = OrganSeg(task=args.task, split=args.split_unlabeled, unlabeled=True, num_cls=config.num_cls, transform=train_transform)
    db_eval = OrganSeg(task=args.task, split=args.split_eval, pre_load=True, num_cls=config.num_cls, transform=eval_transform)

    def worker_init_fn(worker_id):
        random.seed(SEED + worker_id)

    # 保证 labeled 和 unlabeled loader 长度对齐
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

    # --- 模型与优化器 ---
    model, optimizer = make_model_all()  # Student
    ema_model, _ = make_model_all(ema=True)  # Teacher (EMA)

    model = kaiming_normal_init_weight(model)  # 初始化
    logging.info(optimizer)

    # --- 推理器与损失 ---

    sup_loss_func = make_loss_function(args.sup_loss)  # 监督损失

    amp_grad_scaler = GradScaler()

    # --- 训练循环 ---
    best_eval = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    for epoch_num in range(args.max_epoch + 1):
        loss_list, loss_sup_list, loss_cps_list = [], [], []
        model.train()
        cps_w = get_current_consistency_weight(epoch_num)

        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader), total=len(unlabeled_loader), desc=f"epoch {epoch_num} training"):
            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            tmp_bs = image_l.shape[0]  # 有标签样本数

            with autocast('cuda'):
                # === UAMT 核心：不确定性感知一致性训练 ===
                # 1. 学生模型预测
                outputs = model(image)
                outputs_soft = F.softmax(outputs, dim=1)
                # 3. 不确定性估计 (MC Dropout, T=8)
                T = 8
                _, _, d, w, h = image_u.shape
                ema_model.train()  # 启用Dropout
                preds = []
                for i in range(T):
                    noise = torch.clamp(torch.randn_like(image_u) * 0.1, -0.2, 0.2)
                    noisy_input = image_u + noise
                    with torch.no_grad():
                        pred = ema_model(noisy_input)  # 带Dropout的教师预测
                        preds.append(F.softmax(pred, dim=1).unsqueeze(0))
                preds = torch.cat(preds, dim=0)
                mean_pred = torch.mean(preds, dim=0)  # 用MC均值作为教师的目标输出
                uncertainty = -1.0 * torch.sum(mean_pred * torch.log(mean_pred + 1e-6), dim=1, keepdim=True)
                ema_model.eval()  # 恢复 eval 模式

                # 一致性损失使用MC均值作为目标，与不确定性估计保持一致
                consistency_dist = softmax_mse_loss(outputs[tmp_bs:], mean_pred)

                # 4. 监督损失（仅作用于有标签数据）
                loss_sup = sup_loss_func(outputs[:tmp_bs], label_l)

                # 5. 一致性损失（仅作用于无标签数据，且低不确定性区域）
                threshold = (0.75 + 0.25 * sigmoid_rampup(epoch_num, args.max_epoch)) * np.log(2)
                mask = (uncertainty < threshold).float()  # [B, 1, D, H, W]
                consistency_loss = torch.sum(mask * consistency_dist) / (torch.sum(mask) + 1e-16)

                # 6. 总损失
                loss = loss_sup + cps_w * consistency_loss

            # 反向传播
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

            # 更新教师模型
            update_ema_variables(model, ema_model, alpha=0.99, global_step=epoch_num)

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(consistency_loss.item())

        # --- 日志记录 ---
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/total', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/supervised', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/consistency', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list):.4f}, cps_w: {cps_w:.4f}, lr: {get_lr(optimizer):.6f}')

        # 学习率调度
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)

        # === 2. 验证 (沿用CPS的MONAI滑窗+硬Dice) ===
        if (epoch_num > 0 and epoch_num < 150 and epoch_num % 10 == 0) or (epoch_num >= 150 and epoch_num % 1 == 0):
            dice_list_for_epoch = [[] for _ in range(config.num_cls - 1)]
            model.eval()
            ema_model.eval()  # 验证时也可以用 ema_model，或 ensemble，这里沿用 student model 保持简单

            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"epoch {epoch_num} validation"):
                    image, gt = fetch_data(batch)
                    pred_indices = sliding_window_inference(
                        image=image,
                        model=ema_model,
                        patch_size=config.patch_size,
                        num_classes=config.num_cls,
                        overlap=0.25
                    )
                    gt_indices = gt.squeeze(1).long() # 形状 [B,D,H,W]
            
                    # 2. 直接将GPU上的张量传入新的GPU Dice函数
                    dice_per_class = dice_from_labels_gpu(
                        pred_labels=pred_indices,
                        gt_labels=gt_indices,
                        num_classes=config.num_cls
                    )
                    
                    for i in range(len(dice_per_class)):
                        dice_list_for_epoch[i].append(dice_per_class[i])
                        
                    del image, gt, pred_indices, gt_indices
            # --- 评估结果 ---
            dice_mean_per_class = [np.mean(d) for d in dice_list_for_epoch if d]
            if dice_mean_per_class:
                avg_dice = np.mean(dice_mean_per_class)
                logging.info(f'Validation epoch {epoch_num}, Avg Dice: {avg_dice:.4f}')
                
                # class_names = ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'pancreas', 'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate/uterus']
                class_names= ['liver', 'spleen', 'kidney L', 'kidney R', 'stomach', 'gallbladder', 'esophagus',  'pancreas', 'duodenum', 'colon', 'intestine', 'adrenal', 'rectum', 'bladder', 'femur']
                dice_dict = {name: f"{d:.4f}" for name, d in zip(class_names[:len(dice_mean_per_class)], dice_mean_per_class)}
                logging.info(f'Per-class Dice: {dice_dict}')

                if avg_dice > best_eval:
                    best_eval = avg_dice
                    best_epoch = epoch_num
                    epochs_no_improve = 0
                    save_path = os.path.join(snapshot_path, 'ckpts/best_model.pth')
                    torch.save({'model': model.state_dict(), 'ema_model': ema_model.state_dict()}, save_path)
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