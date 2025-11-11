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
from networks.vnet_dycon import VNet_dycon
from utils import maybe_mkdir, get_lr, fetch_data, poly_lr,sliding_window_inference_dycon, dycon_losses
from utils.new_loss import DC_and_CE_loss, RobustCrossEntropyLoss
from utils.transforms import RandomCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg  # 统一使用 OrganSeg，适配 task
from utils.config import Config
# torch.autograd.set_detect_anomaly(True) # <--- 添加这一行

# --- 命令行参数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='dycon')
parser.add_argument('--exp_name', type=str, default='test')
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

# === DyCon-specific Parameters === #
parser.add_argument('--feature_scaler', type=int, default=2, help='Feature scaling factor for contrastive loss')
parser.add_argument('--gamma', type=float, default=1.0, help='Focusing parameter for hard positives/negatives in FeCL (γ)')
parser.add_argument('--beta_min', type=float, default=0.1, help='Minimum value for entropy weighting (β)')
parser.add_argument('--beta_max', type=float, default=1.0, help='Maximum value for entropy weighting (β)')
parser.add_argument('--s_beta', type=float, default=None, help='If provided, use this static beta for UnCLoss instead of adaptive beta.')
parser.add_argument('--temp', type=float, default=0.8, help='Temperature for contrastive softmax scaling (optimal: 0.6)')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight for supervised loss')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight for unsupervised loss')
parser.add_argument('--use_focal', type=int, default=1, help='Whether to use focal weighting (1 for True, 0 for False)')
parser.add_argument('--use_teacher_loss', type=int, default=1, help='Use teacher-based auxiliary loss (1 for True, 0 for False)')
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
    model = VNet_dycon(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        scale_factor=args.feature_scaler,
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
    consistency_criterion = softmax_mse_loss
    amp_grad_scaler = GradScaler()

    # --- 训练循环 ---
    best_eval = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    uncl_criterion = dycon_losses.UnCLoss()
    fecl_criterion = dycon_losses.FeCLoss(device=f"cuda:0", temperature=args.temp, gamma=args.gamma, use_focal=bool(args.use_focal), rampup_epochs=1500)

    for epoch_num in range(args.max_epoch + 1):
        loss_list, loss_sup_list, loss_cps_list = [], [], []
        model.train()
        ema_model.train()
        cps_w = get_current_consistency_weight(epoch_num)
        if args.s_beta is not None:
            beta = args.s_beta
        else:
            beta = dycon_losses.adaptive_beta(epoch=epoch_num, total_epochs=args.max_epoch, max_beta=args.beta_max, min_beta=args.beta_min)
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader), total=len(unlabeled_loader), desc=f"epoch {epoch_num} training"):
            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            labeled_bs = image_l.shape[0]  # 有标签样本数

            with autocast('cuda'):
                noise = torch.clamp(torch.randn_like(image) * 0.1, -0.2, 0.2)
                ema_inputs = image + noise

                _, stud_logits, stud_features = model(image)
                with torch.no_grad():
                    _, ema_logits, ema_features = ema_model(ema_inputs)
            
                stud_probs = F.softmax(stud_logits, dim=1)
                ema_probs = F.softmax(ema_logits, dim=1)
                
                # Calculate the supervised loss
                loss_sup = sup_loss_func(stud_logits[:labeled_bs], label_l)
                
                B, C, _, _, _ = stud_features.shape
                stud_embedding = stud_features.view(B, C, -1)
                stud_embedding = torch.transpose(stud_embedding, 1, 2) 
                stud_embedding = F.normalize(stud_embedding, dim=-1, eps=1e-8)  

                ema_embedding = ema_features.view(B, C, -1)
                ema_embedding = torch.transpose(ema_embedding, 1, 2)
                ema_embedding = F.normalize(ema_embedding, dim=-1, eps=1e-8)

                with torch.no_grad():
                    # ema_logits 是整个批次的输出, 切片 [labeled_bs:] 获取无标签部分
                    unlabeled_logits = ema_logits[labeled_bs:]
                    # 使用 argmax 获取最可能的类别作为伪标签
                    pseudo_label = torch.argmax(unlabeled_logits, dim=1, keepdim=True)

                # 2. 拼接真实标签和伪标签，形成一个完整的、有意义的标签张量
                # label_l 是有标签数据的真实标签
                full_label = torch.cat([label_l, pseudo_label], dim=0)

                # --- 后续步骤保持不变, 因为它们现在操作的是正确的 full_label ---
                # 3. 使用 full_label 来生成 mask_con
                B = full_label.shape[0] 
                num_classes = config.num_cls
                scale = args.feature_scaler * 4

                # Step 3.1: One-hot 完整标签
                label = full_label.squeeze(1).long()
                onehot = F.one_hot(label, num_classes=num_classes)
                onehot = onehot.permute(0, 4, 1, 2, 3).float()

                # Step 3.2: Pooling 以匹配特征图尺寸
                pooled = F.avg_pool3d(onehot, kernel_size=scale, stride=scale)
                pooled_flat = pooled.view(B, num_classes, -1)

                # Step 3.3: 获取 Patch-wise 标签
                patch_labels = torch.argmax(pooled_flat, dim=1)

                # Step 3.4: Reshape 成最终的 mask
                mask_con = patch_labels.unsqueeze(1) # 形状: [B, 1, N]

                # --- 调用损失函数 (保持不变) ---
                teacher_feat = ema_embedding if args.use_teacher_loss else None
                f_loss = fecl_criterion(feat=stud_embedding,
                                        mask=mask_con,
                                        teacher_feat=teacher_feat,
                                        gambling_uncertainty=None,
                                        epoch=epoch_num)
                u_loss = uncl_criterion(stud_logits, ema_logits, beta)
                consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()
                
                # Gather losses
                loss = args.l_weight * loss_sup + cps_w * consistency_loss + args.u_weight * (f_loss + u_loss)

            # 反向传播
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    pred_indices = sliding_window_inference_dycon(
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
                class_names= ['liver', 'spleen', 'kidney L', 'kidney R', 'stomach', 'gallbladder', 'esophagus',  'pancreas', 'duodenum', 'colon', 'intestine', 'adrenal', 'rectum', 'bladder', 'femur L', 'femur R'] 
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