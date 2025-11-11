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

from networks.vnet import VNet
from utils import maybe_mkdir, get_lr, fetch_data, poly_lr, sliding_window_inference,EMA
from utils.new_loss import DC_and_CE_loss, RobustCrossEntropyLoss
from utils.transforms import RandomCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg
from utils.config import Config

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser()
# ... (参数部分与之前完全相同，为简洁省略)
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='dhc')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-sl', '--split_labeled', type=str, default='labeled_2p')
parser.add_argument('-su', '--split_unlabeled', type=str, default='unlabeled_2p')
parser.add_argument('-se', '--split_eval', type=str, default='valid')
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--cps_loss', type=str, default='wce')
parser.add_argument('--sup_loss', type=str, default='w_ce+dice')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.03)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-w', '--cps_w', type=float, default=0.1)
parser.add_argument('-r', '--cps_rampup', default=True)
parser.add_argument('-cr', '--consistency_rampup', type=float, default=150)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

config = Config(args.task)

# --- 辅助函数区 ---
# 自定义的 get_gaussian_importance_map 和 sliding_window_inference 已被移除

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

def make_loss_function(name, weight=None):
    if name == 'ce': return RobustCrossEntropyLoss()
    elif name == 'wce': return RobustCrossEntropyLoss(weight=weight)
    elif name == 'ce+dice': return DC_and_CE_loss(n_classes =config.num_cls)
    elif name == 'w_ce+dice': return DC_and_CE_loss(n_classes =config.num_cls,w_dc=weight, w_ce=weight)
    else: raise ValueError(name)

def make_model_all():
    model = VNet(
        n_channels=config.num_channels, n_classes=config.num_cls, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=True
    ).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=3e-5, nesterov=True)
    return model, optimizer

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
    # print(y_onehot.size())

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)
    
    return tp, fp, fn, tn

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, apply_nonlin=None, batch_dice=True, do_bg=False, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.weight = weight

    def forward(self, x, y, loss_mask=None, is_training=True):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        
        if self.weight is not None: # <--
            if not self.do_bg and self.batch_dice:
                dc *= self.weight[1:]
            else:
                raise NotImplementedError
        
        if not is_training:
            return dc
        else:
            return -dc.mean()

# --- DHC/DistDW 适配 (遵从原接口的最小改动最终版) ---
class DistDW:
    def __init__(self, num_cls, momentum=0.95):
        self.num_cls = num_cls
        self.momentum = momentum
        # 1. 在 init 中声明一个用于 EMA 平滑的类别计数的变量 (NumPy array)
        self.ema_class_counts = np.zeros(num_cls, dtype=np.float32)
        # self.weights 用于存储最近一次计算的权重
        self.weights = torch.ones(num_cls, dtype=torch.float32).cuda() * num_cls

    def _cal_weights(self, num_each_class_np):
        # 保持原样：接收 NumPy 数组，返回 PyTorch Tensor
        num_each_class_t = torch.from_numpy(num_each_class_np).float().cuda()
        P = (num_each_class_t.max() + 1e-8) / (num_each_class_t + 1e-8)
        P_log = torch.log(P)
        weight = P_log / (P_log.max() + 1e-8)
        return weight

    def init_weights(self, labeled_dataset):
        # 计算有标签数据的精确类别分布
        num_each_class_np = np.zeros(self.num_cls)
        for i in tqdm(range(len(labeled_dataset)), desc="Analyzing labeled data for initial distribution"):
            sample = labeled_dataset[i]
            label = sample['label'].numpy().reshape(-1)
            tmp, _ = np.histogram(label, bins=self.num_cls, range=(0, self.num_cls))
            num_each_class_np += tmp
        
        # 2. 使用有标签数据的分布来初始化 EMA 计数器 (NumPy array)
        self.ema_class_counts = num_each_class_np.astype(np.float32)
        
        # 计算并存储初始权重
        initial_weights = self._cal_weights(self.ema_class_counts) * self.num_cls
        self.weights = initial_weights
        
        return self.weights.clone().cpu().numpy()

    def get_ema_weights(self, pseudo_label_logits):
        # 3. 核心修改：EMA 更新类别计数，然后基于平滑后的计数来计算权重
        pseudo_label = torch.argmax(pseudo_label_logits.detach(), dim=1)
        label_numpy = pseudo_label.cpu().numpy()
        
        # a. 计算当前 batch 伪标签的类别计数 (NumPy array)
        batch_class_counts_np = np.zeros(self.num_cls, dtype=np.float32)
        for i in range(label_numpy.shape[0]):
            label = label_numpy[i].flatten()
            tmp, _ = np.histogram(label, bins=self.num_cls, range=(0, self.num_cls))
            batch_class_counts_np += tmp

        # b. 使用 EMA 更新全局的类别计数 (在 NumPy 上操作)
        self.ema_class_counts = (1 - self.momentum) * batch_class_counts_np + self.momentum * self.ema_class_counts
        
        # c. 基于平滑后的全局计数来计算当前权重
        final_weights = self._cal_weights(self.ema_class_counts) * self.num_cls
        self.weights = final_weights # 更新内部权重状态
        
        return self.weights



class DiffDW:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(config.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred,  label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice>0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice<=0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        self.last_dice = cur_dice
        self.cls_learn = EMA(cur_cls_learn, self.cls_learn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)
        cur_diff = torch.pow(cur_diff, 1/5)
        self.dice_weight = EMA(1. - cur_dice, self.dice_weight, momentum=(self.accumulate_iters-1)/self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        weights = weights / weights.max()
        return weights * self.num_cls

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
    logging.info(optimizer_A)
    
    ensemble_model = ModelEnsemble(model_A, model_B)

    diffdw = DiffDW(config.num_cls, accumulate_iters=50)
    distdw = DistDW(config.num_cls, momentum=0.99)

    weight_A = diffdw.init_weights()
    weight_B = distdw.init_weights(labeled_loader.dataset)

    loss_func_A     = make_loss_function(args.sup_loss, weight_A)
    loss_func_B     = make_loss_function(args.sup_loss, weight_B)
    cps_loss_func_A = make_loss_function(args.cps_loss, weight_A)
    cps_loss_func_B = make_loss_function(args.cps_loss, weight_B)
    amp_grad_scaler = GradScaler()

    # --- 主训练循环 ---
    # ... (训练循环的训练部分与之前完全相同，为简洁省略)
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
                output_A = model_A(image)
                output_B = model_B(image)
                del image
                output_A_l, output_A_u = torch.split(output_A, [image_l.shape[0], image_u.shape[0]], dim=0)
                output_B_l, output_B_u = torch.split(output_B, [image_l.shape[0], image_u.shape[0]], dim=0)
                max_A = torch.argmax(output_A.detach(), dim=1, keepdim=True).long()
                max_B = torch.argmax(output_B.detach(), dim=1, keepdim=True).long()

                weight_A = diffdw.cal_weights(output_A_l.detach(), label_l.detach())
                weight_B = distdw.get_ema_weights(output_B_u.detach())

                loss_func_A.update_weight(weight_A,weight_A)
                loss_func_B.update_weight(weight_B,weight_B)
                cps_loss_func_A.update_weight(weight_A)
                cps_loss_func_B.update_weight(weight_B)

                loss_sup = loss_func_A(output_A_l, label_l) + loss_func_B(output_B_l, label_l)
                loss_cps = cps_loss_func_A(output_A, max_B) + cps_loss_func_B(output_B, max_A)
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
                
            #   class_names= ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'pancreas', 'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate/uterus']
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

                if epochs_no_improve >= config.early_stop_patience:
                    logging.info(f'--- Early stopping triggered after {epochs_no_improve} validation cycles with no improvement. ---')
                    break # 跳出主训练循环
    
    logging.info("--- Training Finished ---")
    writer.close()