import tempfile
print(tempfile.gettempdir())
import os
import sys
import logging
import random
from tqdm import tqdm
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

# 假设 bezier_curve.py 和 ramps.py 在utils目录下或PYTHONPATH中
from networks.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, poly_lr, sliding_window_inference
from utils.new_loss import DC_and_CE_loss
from utils.transforms import RandomCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg
from utils.config import Config
try:
    from scipy.special import comb
except:
    from scipy.misc import comb


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

# --- 命令行参数解析 (与新pipeline对齐, 添加算法特定参数) ---
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='slcnet')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('-s', '--seed', type=int, default=1337)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_2p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_2p')
parser.add_argument('-se', '--split_eval', type=str, default='valid')
parser.add_argument('--max_epoch', type=int, default=500) # 按epoch迭代
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--labeled_bs_ratio', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='0')
# --- 算法特定参数 (移植自原代码) ---
parser.add_argument('--consistency', type=float, default=0.1, help='consistency loss weight')
parser.add_argument('--consistency_rampup', type=float, default=150.0, help='consistency rampup in epochs')
parser.add_argument('--model2_inchns', type=int, default=17, help='model2 input channels (1 image + num_classes)')
parser.add_argument('--uncertainty_T', type=int, default=8, help='Number of samples for uncertainty estimation')
parser.add_argument('--use_block_dice', action='store_true',
                    default=False, help='use_block_dice_loss')
parser.add_argument('--block_num', type=int, default=4, help='Number of blocks in each dimension for Block Dice Loss')
args = parser.parse_args()
args.labeled_bs = args.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
config = Config(args.task)
args.model2_inchns = 1 + config.num_cls

# --- 算法核心辅助函数 (移植自原代码) ---
def nonlinear_transformation(slices, flag=True):
    # [忠实实现] 此函数完整复制自原始代码
    if flag:
        random_num = random.random()
        if random_num <= 0.4: return (slices + 1) / 2
        if random_num > 0.4 and random_num <= 0.7:
            points = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
        else:
            points = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    else:
        random_num = random.random()
        if random_num <= 0.4:
            points = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
        elif random_num > 0.4 and random_num <= 0.7:
            points = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
        else:
            points = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
            
    xvals, yvals = bezier_curve(points, nTimes=10000)
    xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_slices = np.interp(slices, xvals, yvals)
    if not flag:
        nonlinear_slices[nonlinear_slices == 1] = -1
    return (nonlinear_slices + 1) / 2

# --- 框架辅助函数 ---
def dice_from_labels_gpu(pred_labels: torch.Tensor, gt_labels: torch.Tensor, num_classes: int):
    # ... (与 baseline 相同) ...
    assert pred_labels.device == gt_labels.device; pred = pred_labels.reshape(-1); gt = gt_labels.reshape(-1)
    dices = []; eps = 1e-8
    for c in range(1, num_classes):
        pred_c = (pred == c); gt_c = (gt == c); intersection = (pred_c & gt_c).sum()
        denominator = pred_c.sum() + gt_c.sum()
        if denominator == 0: dices.append(1.0)
        else: dices.append(((2.0 * intersection) / (denominator + eps)).item())
    return dices

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            # 适用于5D张量 [B, 1, D, H, W]
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target 形状应为 [B, 1, D, H, W]
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

class Block_DiceLoss3D(nn.Module):
    def __init__(self, n_classes, block_num):
        super(Block_DiceLoss3D, self).__init__()
        self.n_classes = n_classes
        self.block_num = block_num
        self.dice_loss = DiceLoss(self.n_classes)

    def forward(self, inputs, target):
        # inputs: [B, C, D, H, W], target: [B, 1, D, H, W]
        shape = inputs.shape
        B, C, D, H, W = shape
        
        # 计算每个维度上的块大小
        block_d = math.ceil(D / self.block_num)
        block_h = math.ceil(H / self.block_num)
        block_w = math.ceil(W / self.block_num)
        
        loss_list = []
        # 在 D, H, W 三个维度上进行循环分块
        for k in range(self.block_num):
            for i in range(self.block_num):
                for j in range(self.block_num):
                    # 对3D体积进行切片
                    d_start, d_end = k * block_d, min((k + 1) * block_d, D)
                    h_start, h_end = i * block_h, min((i + 1) * block_h, H)
                    w_start, w_end = j * block_w, min((j + 1) * block_w, W)

                    block_features = inputs[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    block_labels = target[:, :, d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # block_labels 已经是 [B, 1, d, h, w] 形状，可以直接传入
                    tmp_loss = self.dice_loss(block_features, block_labels)
                    loss_list.append(tmp_loss)
        
        # 对所有块的损失取平均
        return torch.stack(loss_list).mean()

class ModelEnsemble(nn.Module):
    def __init__(self, model_A, model_B):
        super(ModelEnsemble, self).__init__(); self.model_A = model_A; self.model_B = model_B
    def forward(self, x):
        out_A = self.model_A(x); out_soft_A = F.softmax(out_A, dim=1)
        # [忠实实现] 验证时也使用 1 - soft_A 作为输入
        in_B = torch.cat([x, 1 - out_soft_A], dim=1)
        out_B = self.model_B(in_B)
        return (out_soft_A + F.softmax(out_B, dim=1)) / 2.0

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    def forward(self, input, target, mask=None):
        loss = self.ce_loss(input, target)
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
        return loss

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0: return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))

def make_model(in_channels, num_classes):
    model = VNet(
        n_channels=in_channels, n_classes=num_classes, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=True).cuda()
    return model

# --- main ---
if __name__ == '__main__':
    # ... (初始化和日志设置) ...
    SEED=args.seed; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    # snapshot_path = f'./logs/{args.task}'+'10p'+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    snapshot_path = f'./logs/{args.task}'+args.split_labeled[-2:]+f'/{args.exp}/{args.exp_name}/seed_{SEED}'
    maybe_mkdir(snapshot_path); maybe_mkdir(os.path.join(snapshot_path, 'ckpts'))
    writer = SummaryWriter(os.path.join(snapshot_path, 'tensorboard'))
    logging.basicConfig(filename=os.path.join(snapshot_path, 'train.log'), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)); logging.info(str(args))

    # === 1. 数据加载 (使用新管线方式) ===
    train_transform = transforms.Compose([RandomCrop(config.patch_size), RandomFlip_LR(), RandomFlip_UD(), ToTensor()])
    db_labeled = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=train_transform)
    db_unlabeled = OrganSeg(task=args.task, split=args.split_unlabeled, unlabeled=True, num_cls=config.num_cls, transform=train_transform)
    db_eval = OrganSeg(task=args.task, split=args.split_eval, pre_load=True, num_cls=config.num_cls, transform=transforms.Compose([ToTensor()]))
    
    def worker_init_fn(worker_id):
        random.seed(SEED + worker_id)
    
    sampler = torch.utils.data.RandomSampler(db_labeled, replacement=True, num_samples=len(db_unlabeled))
    labeled_loader = DataLoader(
        db_labeled, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True
    )
    unlabeled_loader = DataLoader(
        db_unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True
    )
    eval_loader = DataLoader(db_eval, batch_size=1, shuffle=False, pin_memory=True)
    logging.info(f'{len(unlabeled_loader)} iterations per epoch')
    
    # --- 模型、优化器和损失函数设置 ---
    model1 = make_model(in_channels=1, num_classes=config.num_cls)
    model2 = make_model(in_channels=args.model2_inchns, num_classes=config.num_cls)
    ensemble_model = ModelEnsemble(model1, model2)

    optimizer1 = optim.SGD(model1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    
    sup_ce_loss = nn.CrossEntropyLoss()
    if args.use_block_dice:
        # 如果使用 block dice, sup_dice_loss 就是 Block_DiceLoss3D 的实例
        sup_dice_loss = Block_DiceLoss3D(n_classes=config.num_cls, block_num=args.block_num)
        logging.info(f"Using Block Dice Loss with {args.block_num}x{args.block_num}x{args.block_num} blocks.")
    else:
        # 否则，使用我们新管线中标准的Dice Loss
        # 假设 DC_and_CE_loss 有一个 dice_only 模式
        sup_dice_loss = DiceLoss(n_classes=config.num_cls) 

    cps_loss_func = MaskedCrossEntropyLoss()
    amp_grad_scaler = GradScaler()
    
    # --- 主训练循环 ---
    best_eval = 0.0
    iter_num = 0
    max_iterations = args.max_epoch * len(unlabeled_loader)

    for epoch_num in range(args.max_epoch):
        model1.train()
        model2.train()
        for i_batch, (labeled_sample, unlabeled_sample) in tqdm(enumerate(zip(labeled_loader, unlabeled_loader)), total=len(unlabeled_loader), desc=f"Epoch {epoch_num}"):
            image_l, label_l = fetch_data(labeled_sample)
            image_u = fetch_data(unlabeled_sample, labeled=False)
            volume_batch = torch.cat([image_l, image_u], dim=0)

            # 1. 准备输入
            volume_batch_model1 = torch.zeros_like(volume_batch)
            volume_batch_model2_img = torch.zeros_like(volume_batch)
            for i in range(volume_batch.shape[0]):
                slices_norm = volume_batch[i, 0].cpu().numpy() * 2 - 1
                volume_batch_model1[i, 0] = torch.from_numpy(nonlinear_transformation(slices_norm, True)).cuda()
                volume_batch_model2_img[i, 0] = torch.from_numpy(nonlinear_transformation(slices_norm, False)).cuda()
            
            with autocast('cuda'):
                # 2. 前向传播
                outputs1 = model1(volume_batch_model1)
                outputs_soft1 = F.softmax(outputs1, dim=1)
                
                # [修正] 严格遵循原代码的噪声范围
                noise = torch.clamp(torch.randn_like(volume_batch_model2_img) * 0.1, -0.05, 0.05)
                # [修正] 严格遵循原代码使用 1 - outputs_soft1 作为输入
                model2_input = torch.cat((volume_batch_model2_img + noise, 1 - outputs_soft1.detach()), dim=1)
                outputs2 = model2(model2_input)
                outputs_soft2 = F.softmax(outputs2, dim=1)

                if args.use_block_dice:
                    # 使用 Block Dice
                    loss_block_dice1 = sup_dice_loss(outputs_soft1[:args.labeled_bs], label_l)
                    loss1_sup = 0.5 * (sup_ce_loss(outputs1[:args.labeled_bs], label_l.squeeze(1).long()) + loss_block_dice1)
                    
                    loss_block_dice2 = sup_dice_loss(outputs_soft2[:args.labeled_bs], label_l)
                    loss2_sup = 0.5 * (sup_ce_loss(outputs2[:args.labeled_bs], label_l.squeeze(1).long()) + loss_block_dice2)
                else:
                    # 使用标准的 Dice
                    loss1_sup = 0.5 * (sup_ce_loss(outputs1[:args.labeled_bs], label_l.squeeze(1).long()) + sup_dice_loss(outputs_soft1[:args.labeled_bs], label_l))
                    loss2_sup = 0.5 * (sup_ce_loss(outputs2[:args.labeled_bs], label_l.squeeze(1).long()) + sup_dice_loss(outputs_soft2[:args.labeled_bs], label_l))
                # 4. 计算一致性损失
                consistency_weight = args.consistency * sigmoid_rampup(epoch_num, args.consistency_rampup)

                # [忠实实现] 不确定性估计与掩码
                unlabeled_input_m1 = volume_batch_model1[args.labeled_bs:]
                stride = unlabeled_input_m1.shape[0]
                preds = torch.zeros([stride * args.uncertainty_T, config.num_cls] + list(config.patch_size), device='cuda')

                with torch.no_grad():
                    for i in range(args.uncertainty_T // 2):
                        # [修正] 严格遵循原代码的噪声范围
                        ema_inputs = unlabeled_input_m1.repeat(2, 1, 1, 1, 1) + torch.clamp(torch.randn_like(unlabeled_input_m1.repeat(2, 1, 1, 1, 1)) * 0.1, -0.05, 0.05)
                        preds[2 * stride * i : 2 * stride * (i + 1)] = model1(ema_inputs)
                
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(args.uncertainty_T, stride, config.num_cls, *config.patch_size)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
                
                threshold = (0.75 + 0.25 * sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
                mask = (uncertainty < threshold).float()
                
                pseudo_outputs1 = torch.argmax(outputs_soft1[args.labeled_bs:].detach(), dim=1)
                pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1)
                # [忠实实现] squeeze mask to match loss dimension
                pseudo_supervision2 = cps_loss_func(outputs2[args.labeled_bs:], pseudo_outputs1, mask.squeeze(1))
                
                pseudo_supervision1 = cps_loss_func(outputs1[args.labeled_bs:], pseudo_outputs2)

                # 5. 组合总损失
                model1_loss = loss1_sup + consistency_weight * pseudo_supervision1
                model2_loss = loss2_sup + consistency_weight * pseudo_supervision2
                loss = model1_loss + model2_loss
            
            # --- 反向传播 ---
            optimizer1.zero_grad(); optimizer2.zero_grad()
            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer1); amp_grad_scaler.step(optimizer2)
            amp_grad_scaler.update()

            iter_num += 1
            # [修正] 按 iteration 更新学习率
            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups: param_group['lr'] = lr_
            for param_group in optimizer2.param_groups: param_group['lr'] = lr_
            
            # ... (日志记录) ...
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/total', loss.item(), iter_num)

        # === 2. 验证循环 (使用新管线) ===
        if (epoch_num > 0 and epoch_num <150 and epoch_num % 10 == 0) or (epoch_num >= 150 and epoch_num % 1 == 0):
            model1.eval(); model2.eval()
            dice_list_for_epoch = [[] for _ in range(config.num_cls - 1)]
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
            
            # --- 结果聚合与保存 ---
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
                    torch.save({'A': model1.state_dict(), 'B': model2.state_dict()}, save_path)
                    logging.info(f'New best model saved to {save_path}')
                else:
                        epochs_no_improve += 1
                logging.info(f'\t Best eval dice is {best_eval:.4f} in epoch {best_epoch}')
                logging.info(f'\t Epochs with no improvement: {epochs_no_improve} / {config.early_stop_patience}')

                if epochs_no_improve >= config.early_stop_patience:
                    logging.info(f'--- Early stopping triggered after {epochs_no_improve} validation cycles with no improvement. ---')
                    break # 跳出主训练循环
    
    writer.close()
    logging.info("--- Training Finished ---")