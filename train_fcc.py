# dar is short for distribution aware reweighting
# 该脚本实现了基于分布感知重加权的半监督
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

# --- MONAI ---
from monai.inferers import SlidingWindowInferer

from networks.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, calculate_class_distribution, EMA, sliding_window_inference
from utils.new_loss import DC_and_CE_loss, RobustCrossEntropyLoss
from utils.transforms import RandomCrop, CenterCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg
from utils.config import Config
import math

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser()
# ... (参数部分与之前完全相同，为简洁省略)
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='fcc')
parser.add_argument('--exp_name', type=str, default='base')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_2p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_2p')
parser.add_argument('-se', '--split_eval', type=str, default='valid')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--start_epoch', type=int, default=100)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('--weight_floor', type=float, default=0.05)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=150)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)
def calculate_entropy_map(probs_tensor):
    """计算一个softmax概率张量的逐像素熵。"""
    epsilon = 1e-10
    return -torch.sum(probs_tensor * torch.log(probs_tensor + epsilon), dim=1)

def get_sorted_positions_torch(class_counts: torch.Tensor, decending=True) -> torch.Tensor:
    """处理Tensor输入的get_sorted_positions版本。"""
    sorted_indices = torch.argsort(class_counts, descending=decending)
    ranks = torch.empty_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(class_counts), device=class_counts.device)
    return ranks

# 您可以创建一个新的函数或直接替换旧的函数
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

class FCC(nn.Module):
    """
    """
    def __init__(self, num_classes, strength=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.strength = strength         # gamma缩放的整体强度

    def forward(self, features, pseudo_labels, global_class_ranks):
        # 2. 计算 gamma 缩放因子
        with torch.no_grad():    
            inter_class_term = 1 - global_class_ranks / (self.num_classes - 1)
            inter_class_map = inter_class_term[pseudo_labels.squeeze(1).long()]
            gamma_map = 1.0 + self.strength * inter_class_map
            gamma_map = gamma_map.unsqueeze(1)

        scaled_features = features * gamma_map
        return scaled_features

class DistributionAwareReweighting:
    """
    最终版、高度稳健的分布感知动态权重计算器 (DAW-EMA)。
    
    融合了所有最佳实践：
    - 在概率（比例）而非计数上进行EMA。
    - 采用您新提出的、更精确的对数域归一化公式。
    - 使用百分位点和钳位来抵抗极端离群点。
    - （可选）为CE和Dice提供解耦的权重。
    """
    def __init__(self, num_cls, momentum=0.99, device='cuda', bg_index=0, cap_ratio=8.0, 
                 use_dice_weight=False, weight_ceiling=1.0, weight_floor=0.05):
        self.num_cls = num_cls
        self.momentum = momentum
        self.weight_ceiling = weight_ceiling
        self.device = device
        self.bg_index = bg_index
        self.cap_ratio = cap_ratio
        self.use_dice_weight = use_dice_weight # 控制是否为Dice生成权重
        self.weight_floor = weight_floor
        # global_dist 存储的是平滑后的概率分布
        self.global_dist = torch.zeros(self.num_cls, device=self.device)
        self.initialized = False

    def initialize(self, labeled_dataset):
        """在训练开始前，根据有标签数据初始化全局分布"""
        print("Initializing DistributionAwareReweighting module...")
        total_counts = torch.zeros(self.num_cls, device=self.device)
        
        for i in tqdm(range(len(labeled_dataset)), desc="Analyzing labeled data for initial distribution"):
            sample = labeled_dataset[i]
            label_tensor = torch.from_numpy(sample['label']).to(self.device)
            total_counts += calculate_class_distribution(label_tensor, self.num_cls)
            
        # 归一化为概率分布
        if total_counts.sum() > 0:
            self.global_dist = total_counts / total_counts.sum()
        else: # 避免除以0
            self.global_dist.fill_(1.0 / self.num_cls)
            
        self.initialized = True
        print("Initialization complete.")
        print(f"Initial global distribution (probabilities): {self.global_dist.cpu().numpy()}")

    def _calculate_weights(self, distribution):
        """
        核心权重计算函数：采纳了您最新的公式，并结合了鲁棒性设计。
        """
        p = distribution
        # 1. 只在前景类别中计算百分位点
        fg_indices = [i for i in range(self.num_cls) if i != self.bg_index]
        p_foreground = p[fg_indices]
        p_foreground = p_foreground[p_foreground > 0] # 只考虑存在的

        if len(p_foreground) < 2:
            return torch.ones_like(p), torch.ones_like(p)

        # 2. 计算稳健的范围 (robust range)
        robust_min = torch.quantile(p_foreground, 0.05)
        robust_max = torch.quantile(p_foreground, 0.95)

        if robust_max - robust_min < 1e-8:
            return torch.ones_like(p), torch.ones_like(p)

        # 3. 将原始概率分布钳位到稳健范围内
        p_clipped = torch.clamp(p, min=robust_min, max=robust_max)
        
        # 4. 应用您提出的对数域归一化公式
        epsilon = 1e-8
        log_p = torch.log(p_clipped + epsilon)
        log_min = torch.log(robust_min + epsilon)
        log_max = torch.log(robust_max + epsilon)
        
        # 归一化到 [0, 1] 区间
        p_normalized_log = (log_p - log_min) / (log_max - log_min + epsilon)
        # 5. 计算最终权重并进行后处理
        w_ce = self.weight_floor + (self.weight_ceiling - self.weight_floor) * (1 -p_normalized_log)
        
        # a. 背景权重特殊处理：可以设为1.0或前景中位数值
        fg_weights = w_ce[fg_indices]
        w_ce[self.bg_index] = torch.median(fg_weights)
        
        # b. 权重上限 (Capping)
        median_fg_weight = torch.median(fg_weights)
        w_ce = torch.clamp(w_ce, max=median_fg_weight * self.cap_ratio)
        
        # c. 权重均值归一化，稳定学习率
        w_ce = w_ce / w_ce.mean()
        
        # d. 为Dice生成温和的权重
        if self.use_dice_weight:
            w_dc = torch.sqrt(w_ce).clamp_max(2.0)
            # w_dc = w_ce
        else:
            w_dc = torch.ones_like(w_ce)
        # print(f"CE weights: {w_ce.cpu().numpy()}")

        return w_ce, w_dc

    def get_weights(self, pseudo_label_logits):
        """在每个训练迭代中调用此函数"""
        if not self.initialized:
            raise RuntimeError("Module has not been initialized.")
            
        # 1. 计算当前batch的伪标签概率分布
        probs = torch.softmax(pseudo_label_logits.detach(), dim=1)
        batch_sum = probs.sum(dim=(0, 2, 3, 4))
        if batch_sum.sum() == 0: # 如果batch中全是背景
            batch_prob = torch.zeros_like(self.global_dist)
        else:
            batch_prob = batch_sum / batch_sum.sum()
        
        # 2. EMA平滑更新全局概率分布
        self.global_dist = EMA(batch_prob, self.global_dist, self.momentum)
        
        # 3. 基于更新后的全局概率分布，计算CE和Dice的权重
        return self._calculate_weights(self.global_dist)

class ModelEnsemble(nn.Module):
    def __init__(self, model_A, model_B):
        super(ModelEnsemble, self).__init__()
        self.model_A = model_A
        self.model_B = model_B

    def forward(self, x):
        return (self.model_A(x) + self.model_B(x)) / 2.0

# ... (其他辅助函数如 sigmoid_rampup, kaiming_normal_init_weight 等与之前相同，为简洁省略)
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

def get_current_uncertainty_weight(epoch,start_epoch):
    if epoch < start_epoch:
        return 0.0
    else:
        rampup_len = args.uncertainty_rampup if args.uncertainty_rampup is not None else args.max_epoch
        return args.unc_w * sigmoid_rampup(epoch-start_epoch, rampup_len)

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def make_model_all():
    model = VNet(
        n_channels=config.num_channels, n_classes=config.num_cls, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=True
    ).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    return model, optimizer

if __name__ == '__main__':
    # --- 初始化和日志设置 ---
    SEED=args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # snapshot_path = f'./logs/{args.task}'+'10p'+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    snapshot_path = f'./logs/{args.task}'+args.split_labeled[-2:]+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    # ... (日志和文件夹创建部分与之前相同)
    maybe_mkdir(snapshot_path)
    maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(
        filename=os.path.join(snapshot_path, 'train.log'), level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # === 1. 数据加载 (已更新为统一方式) ===
    # ... (数据加载部分与之前修改后的一样，为简洁省略)
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
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble_model = ModelEnsemble(model_A, model_B)
    db_labeled_for_analysis = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=None)
    daw_rewighter = DistributionAwareReweighting(num_cls=config.num_cls, momentum=0.99, use_dice_weight=True,weight_floor=args.weight_floor)
    daw_rewighter.initialize(db_labeled_for_analysis)
    feature_dim = config.n_filters
    fcc_module_A = FCC(num_classes=config.num_cls).cuda()
    fcc_module_B = FCC(num_classes=config.num_cls).cuda()
    logging.info(optimizer_A)

    loss_func_A = DC_and_CE_loss(n_classes=config.num_cls)
    loss_func_B = DC_and_CE_loss(n_classes=config.num_cls)
    cps_loss_func_A = RobustCrossEntropyLoss()
    cps_loss_func_B = RobustCrossEntropyLoss()
    amp_grad_scaler = GradScaler()

    # --- 主训练循环 ---
    best_eval = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list, loss_sup_list, loss_cps_list = [], [], []
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
                features_A, initial_logits_A = model_A(image, return_features=True)
                features_B, initial_logits_B = model_B(image, return_features=True)
                del image
                if epoch_num >= args.start_epoch:
                    with torch.no_grad():
                        # --- 准备 AFR 的输入 ---
                        probs_A = torch.softmax(initial_logits_A, dim=1)
                        probs_B = torch.softmax(initial_logits_B, dim=1)
                        pseudo_A = torch.argmax(initial_logits_A, dim=1, keepdim=True).long()
                        pseudo_B = torch.argmax(initial_logits_B, dim=1, keepdim=True).long()
            
                        current_global_dist = daw_rewighter.global_dist
                        global_class_ranks = get_sorted_positions_torch(current_global_dist)

                    regularized_features_A = fcc_module_A(features_A, pseudo_A, global_class_ranks)
                    regularized_features_B = fcc_module_B(features_B, pseudo_B, global_class_ranks)
                    # --- 最终预测 ---
                    final_logits_A = model_A.out_conv(regularized_features_A)
                    final_logits_B = model_B.out_conv(regularized_features_B)
                else:
                    final_logits_A = initial_logits_A
                    final_logits_B = initial_logits_B
                
                final_logits_A_l, final_logits_A_u = torch.split(final_logits_A, [image_l.shape[0], image_u.shape[0]], dim=0)
                final_logits_B_l, final_logits_B_u = torch.split(final_logits_B, [image_l.shape[0], image_u.shape[0]], dim=0)
                
                # b. DAW模块必须基于“最终预测”来更新
                wA_ce, wA_dc = daw_rewighter.get_weights(final_logits_A_u)
                wB_ce, wB_dc = daw_rewighter.get_weights(final_logits_B_u)

                # --- 后续损失计算 (使用同一份 output_A, output_B) ---              
                loss_sup = loss_func_A(final_logits_A_l, label_l) + loss_func_B(final_logits_B_l, label_l)
                
                with torch.no_grad():
                    # 伪标签也应从最终预测中产生
                    final_pseudo_B = torch.argmax(final_logits_B, dim=1, keepdim=True).long()
                    final_pseudo_A = torch.argmax(final_logits_A, dim=1, keepdim=True).long()
                
                loss_cps = cps_loss_func_A(final_logits_A, final_pseudo_B) + cps_loss_func_B(final_logits_B, final_pseudo_A)
                loss = loss_sup + cps_w * loss_cps

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer_A)
            amp_grad_scaler.step(optimizer_B)
            amp_grad_scaler.update()
            loss_list.append(loss.item())
            loss_sup_list.append(loss_sup.item())
            loss_cps_list.append(loss_cps.item())
        
        writer.add_scalar('lr', get_lr(optimizer_A), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/total', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/supervised', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/consistency', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list):.4f}, cps_w: {cps_w:.4f}, lr: {get_lr(optimizer_A):.6f}')
        
        optimizer_A.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        optimizer_B.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        
        # === 2. 验证循环 (已更新为 MONAI 滑窗推理) ===
        if (epoch_num > 0 and epoch_num <150 and epoch_num % 10 == 0) or (epoch_num >= 150 and epoch_num % 1 == 0):
        # if epoch_num >= 0 and epoch_num % 1 == 0:
            dice_list_for_epoch = [[] for _ in range(config.num_cls - 1)]
            model_A.eval()
            model_B.eval()

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
                    # output_logits = inferer(inputs=image, network=ensemble_model)
                    # pred_indices = torch.argmax(output_logits, dim=1).long()
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
            # --- 结果聚合与保存逻辑 (与之前相同) ---
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
                    break # 跳出主训练循环
    
    logging.info("--- Training Finished ---")
    writer.close()