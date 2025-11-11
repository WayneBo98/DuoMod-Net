import tempfile
print(tempfile.gettempdir())
import os
import sys
import logging
import random
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from networks.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, sliding_window_inference
from utils.new_loss import DC_and_CE_loss, DiceLoss # 假设 SoftDiceLoss 在这里
from utils.transforms import RandomCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg
from utils.config import Config

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='dmd') # 实验名称
parser.add_argument('--exp_name', type=str, default='test2')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_5p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_5p')
parser.add_argument('-se', '--split_eval', type=str, default='valid')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='0')
# --- DMD 适配: 添加新参数 ---
parser.add_argument('--lam_kd', type=float, default=0.1, help='Weight for distillation loss')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for distillation')
parser.add_argument('--kd_rampup', type=float, default=150.0, help='Rampup length for distillation loss weight')
# --- End DMD 适配 ---

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)

# --- 辅助函数区 ---
# ... (dice_from_labels_gpu, ModelEnsemble, kaiming/xavier_init 与 baseline 完全相同, 为简洁省略)
def dice_from_labels_gpu(pred_labels: torch.Tensor, gt_labels: torch.Tensor, num_classes: int):
    assert pred_labels.device == gt_labels.device; pred = pred_labels.reshape(-1); gt = gt_labels.reshape(-1)
    dices = []; eps = 1e-8
    for c in range(1, num_classes):
        pred_c = (pred == c); gt_c = (gt == c); intersection = (pred_c & gt_c).sum()
        denominator = pred_c.sum() + gt_c.sum()
        if denominator == 0: dices.append(1.0)
        else: dices.append(((2.0 * intersection) / (denominator + eps)).item())
    return dices

class ModelEnsemble(nn.Module):
    def __init__(self, model_A, model_B): super(ModelEnsemble, self).__init__(); self.model_A = model_A; self.model_B = model_B
    def forward(self, x): return (self.model_A(x) + self.model_B(x)) / 2.0
    
def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d): torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d): m.weight.data.fill_(1); m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d): torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d): m.weight.data.fill_(1); m.bias.data.zero_()
    return model

# --- DMD 适配: rampup 函数 ---
def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0: return 1.0
    current = np.clip(current, 0.0, rampup_length); phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def get_current_kd_weight(epoch):
    if args.kd_rampup == 0:
        return args.lam_kd
    return args.lam_kd * sigmoid_rampup(epoch, args.kd_rampup)
# --- End DMD 适配 ---


def make_model_all():
    model = VNet(
        n_channels=config.num_channels, n_classes=config.num_cls, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=True
    ).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    return model, optimizer

if __name__ == '__main__':
    # ... (初始化和日志设置) ...
    SEED=args.seed; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    snapshot_path = f'./logs/{args.task}'+args.split_labeled[-2:]+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    # snapshot_path = f'./logs/{args.task}'+'10p'+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    maybe_mkdir(snapshot_path); maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(filename=os.path.join(snapshot_path, 'train.log'), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)); logging.info(str(args))

    # === 1. 数据加载 ===
    train_transform = transforms.Compose([RandomCrop(config.patch_size), RandomFlip_LR(), RandomFlip_UD(), ToTensor()])
    eval_transform = transforms.Compose([ToTensor()])
    db_labeled = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=train_transform)
    db_unlabeled = OrganSeg(task=args.task, split=args.split_unlabeled, unlabeled=True, num_cls=config.num_cls, transform=train_transform)
    db_eval = OrganSeg(task=args.task, split=args.split_eval, pre_load=True, num_cls=config.num_cls, transform=eval_transform)
    def worker_init_fn(worker_id): random.seed(SEED + worker_id)
    sampler = torch.utils.data.RandomSampler(db_labeled, replacement=True, num_samples=len(db_unlabeled))
    labeled_loader = DataLoader(db_labeled, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabeled_loader = DataLoader(db_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    eval_loader = DataLoader(db_eval, batch_size=1, shuffle=False, pin_memory=True)
    logging.info(f'{len(unlabeled_loader)} iterations per epoch')

    # --- 模型、优化器和损失函数设置 ---
    model_A, optimizer_A = make_model_all()
    model_B, optimizer_B = make_model_all()
    model_A = kaiming_normal_init_weight(model_A)
    model_B = xavier_normal_init_weight(model_B)
    ensemble_model = ModelEnsemble(model_A, model_B)
    
    # --- DMD 适配: 定义损失函数 ---
    sup_loss_func = DC_and_CE_loss(n_classes=config.num_cls)
    # 使用 SoftDiceLoss 作为蒸馏损失
    distill_loss_func = DiceLoss(n_classes=config.num_cls, soft_max=False, target_is_soft=True)
    # --- End DMD 适配 ---
    
    amp_grad_scaler = GradScaler()

    # --- 主训练循环 ---
    best_eval, best_epoch, epochs_no_improve = 0.0, 0, 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list, loss_sup_list, loss_kd_list = [], [], []
        model_A.train(); model_B.train()
        kd_weight = get_current_kd_weight(epoch_num)

        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader), total=len(unlabeled_loader), desc=f"epoch {epoch_num} training"):
            optimizer_A.zero_grad(); optimizer_B.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            
            with autocast('cuda'):
                output_A = model_A(image)
                output_B = model_B(image)
                
                # --- DMD 适配: 核心损失计算逻辑 ---
                
                # 1. 计算有监督损失 (只在有标签数据上)
                output_A_l, _ = torch.split(output_A, [image_l.shape[0], image_u.shape[0]], dim=0)
                output_B_l, _ = torch.split(output_B, [image_l.shape[0], image_u.shape[0]], dim=0)
                loss_sup = sup_loss_func(output_A_l, label_l) + sup_loss_func(output_B_l, label_l)
                
                # 2. 计算温度缩放后的软化概率 (在所有数据上)
                # 注意：这里用 softmax 对应多分类，而不是原文二分类的 sigmoid
                soft_A_distill = torch.softmax(output_A / args.temperature, dim=1)
                soft_B_distill = torch.softmax(output_B / args.temperature, dim=1)

                # 3. 计算 Dice 蒸馏损失 (在所有数据上)
                loss_distill = distill_loss_func(soft_A_distill, soft_B_distill.detach()) + \
                               distill_loss_func(soft_B_distill, soft_A_distill.detach())
                
                # 4. 组合总损失
                loss = loss_sup + kd_weight * loss_distill
                # --- End DMD 适配 ---

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer_A)
            amp_grad_scaler.step(optimizer_B)
            amp_grad_scaler.update()

            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_kd_list.append(loss_distill.item())
        
        # --- 日志和学习率调整 ---
        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('distill_weight', kd_weight, epoch_num)
        writer.add_scalar('loss/total', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/supervised', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/distillation', np.mean(loss_kd_list), epoch_num)

        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list):.4f}, kd_w: {kd_weight:.4f}, lr: {get_lr(optimizer_A):.6f}')
        
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        
        # === 2. 验证循环 (与 baseline 完全相同，直接复用) ===
        if (epoch_num > 0 and epoch_num <150 and epoch_num % 10 == 0) or (epoch_num >= 150 and epoch_num % 1 == 0):
            # ... (验证代码与 CReST/ADSH 版本完全相同, 为简洁省略)
            dice_list_for_epoch = [[] for _ in range(config.num_cls - 1)]
            model_A.eval(); model_B.eval()
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"epoch {epoch_num} validation"):
                    image, gt = fetch_data(batch)
                    pred_indices = sliding_window_inference(
                        image=image,
                        model=ensemble_model,
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
            
            dice_mean_per_class = [np.mean(d) for d in dice_list_for_epoch if d]
            if dice_mean_per_class:
                avg_dice = np.mean(dice_mean_per_class)
                logging.info(f'Validation epoch {epoch_num}, Avg Dice: {avg_dice:.4f}')
                
                # class_names= ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'pancreas', 'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate/uterus']
                class_names= ['liver', 'spleen', 'kidney L', 'kidney R', 'stomach', 'gallbladder', 'esophagus',  'pancreas', 'duodenum', 'colon', 'intestine', 'adrenal', 'rectum', 'bladder', 'femur L', 'femur R'] 
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

                if epochs_no_improve >= 49:
                    logging.info(f'--- Early stopping triggered after {epochs_no_improve} validation cycles with no improvement. ---')
                    break # 跳出主训练循环
    
    
    logging.info("--- Training Finished ---")
    writer.close()