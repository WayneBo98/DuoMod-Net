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
from networks.vnet_skcdf import VNet_Decouple_Attention_ABC
from utils import maybe_mkdir, get_lr, fetch_data, seed_worker, poly_lr, sliding_window_inference_skcdf
from utils.new_loss import DC_and_CE_loss, RobustCrossEntropyLoss
from utils.transforms import RandomCrop, ToTensor, RandomFlip_LR, RandomFlip_UD
from utils.data_loaders import OrganSeg
from utils.config import Config

# --- 命令行参数解析 ---
parser = argparse.ArgumentParser()
# ... (参数部分与之前完全相同，为简洁省略)
parser.add_argument('--task', type=str, default='word')
parser.add_argument('--exp', type=str, default='skcdf')
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
parser.add_argument('--ema_w', type=float, default=0.99)
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
    model = VNet_Decouple_Attention_ABC(
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
    init_transform = transforms.Compose([ToTensor()])
    db_labeled = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=train_transform)
    db_unlabeled = OrganSeg(task=args.task, split=args.split_unlabeled, unlabeled=True, num_cls=config.num_cls, transform=train_transform)
    db_eval = OrganSeg(task=args.task, split=args.split_eval, pre_load=True, num_cls=config.num_cls, transform=eval_transform)
    db_forinition = OrganSeg(task=args.task, split=args.split_labeled, num_cls=config.num_cls, transform=init_transform)
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
    model, optimizer = make_model_all()
    model = xavier_normal_init_weight(model)

    loss_func = make_loss_function(args.sup_loss)
    cps_loss_func = make_loss_function(args.cps_loss)
    amp_grad_scaler = GradScaler()

    num_cls = config.num_cls
    num_sample = 4
    loop = 0
    class_size = [0 for c in range(num_cls)]  # {0: 0, 1: 0, 2: 0, 3: 0,...}
    for sample in db_forinition:
        image = sample["image"]
        label = sample["label"]
        label = label.cpu().numpy()
        for i in range(num_cls):
            num = np.sum(label == i)
            class_size[i] = class_size[i] + num
        loop = loop + 1
        if loop == num_sample:
            break
    # print(class_size)
    ir2 = min(class_size) / (np.array(class_size)+ 1e-8)
    ir2 = torch.tensor(ir2).cuda()
    # --- 主训练循环 ---
    # ... (训练循环的训练部分与之前完全相同，为简洁省略)
    best_eval = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    for epoch_num in range(args.max_epoch + 1):
        loss_list, loss_sup_list, loss_cps_list = [], [], []
        model.train()
        cps_w = get_current_consistency_weight(epoch_num)
        for batch_l, batch_u in tqdm(zip(labeled_loader, unlabeled_loader), total=len(unlabeled_loader), desc=f"epoch {epoch_num} training"):
            
            for out_conv9_name, out_conv9_params in model.decoder_l.out_conv9.named_parameters():
                    if out_conv9_name in model.decoder_l.out_conv9_abc.state_dict().keys():
                        out_conv9_abc_params = model.decoder_l.out_conv9_abc.state_dict()[out_conv9_name]
                        if out_conv9_params.shape == out_conv9_abc_params.shape:
                            out_conv9_params.data = args.ema_w * out_conv9_params.data + (1 - args.ema_w) * out_conv9_abc_params.data

            for decoder_u_name, decoder_u_params in model.decoder_u.named_parameters():
                if decoder_u_name in model.decoder_l.state_dict().keys():
                    decoder_l_params = model.decoder_l.state_dict()[decoder_u_name]
                    if decoder_u_params.shape == decoder_l_params.shape:
                        decoder_u_params.data = args.ema_w * decoder_u_params.data + (1 - args.ema_w) * decoder_l_params.data

            for out_conv9_name, out_conv9_params in model.decoder_u.out_conv9.named_parameters():
                if out_conv9_name in model.decoder_u.out_conv9_abc.state_dict().keys():
                    out_conv9_abc_params = model.decoder_u.out_conv9_abc.state_dict()[out_conv9_name]
                    if out_conv9_params.shape == out_conv9_abc_params.shape:
                        out_conv9_params.data = args.ema_w * out_conv9_params.data + (
                                    1 - args.ema_w) * out_conv9_abc_params.data 
            
            optimizer.zero_grad()
            image_l, label_l = fetch_data(batch_l)
            image_u = fetch_data(batch_u, labeled=False)
            image = torch.cat([image_l, image_u], dim=0)
            with autocast('cuda'):
                output_A,output_A_abc = model(image, pred_type = "labeled" )
                del image
                output_l = output_A[:image_l.shape[0], ...]
                output_l_abc = output_A_abc[:image_l.shape[0], ...]
                
                label_l_prob = label_l.to(dtype=torch.float64).detach()
                for i in range(num_cls):
                    label_l_prob = torch.where(label_l_prob == i, ir2[i], label_l_prob)
                maskforbalance = torch.bernoulli(label_l_prob.detach())

                L_sup = loss_func(output_l,label_l) + loss_func(output_l_abc * maskforbalance,label_l * maskforbalance)

                pseudo_label_prob = output_A[image_l.shape[0]:, ...]
                pseudo_label_prob_abc = output_A_abc[image_l.shape[0]:, ...]

                pseudo_label = torch.argmax(pseudo_label_prob, dim=1, keepdim=True).long()
                pseudo_label_abc = torch.argmax(pseudo_label_prob_abc, dim=1, keepdim=True).long()

                ir22 = 1 - (epoch_num / args.max_epoch) * (1 - ir2)

                pseudo_label_abc_prob = pseudo_label_abc.to(dtype=torch.float64).detach()
                for i in range(num_cls):
                    pseudo_label_abc_prob = torch.where(pseudo_label_abc_prob == i, ir22[i], pseudo_label_abc_prob)
                maskforbalance_u = torch.bernoulli(pseudo_label_abc_prob.detach())

                output_u, output_u_abc = model(image_u, pred_type = "unlabeled")

                L_u = cps_loss_func(output_u, pseudo_label.detach()) + cps_loss_func(output_u_abc * maskforbalance_u, pseudo_label_abc.detach() * maskforbalance_u)

                loss = L_sup + cps_w * L_u

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
            loss_list.append(loss.item())
            loss_sup_list.append(L_sup.item())
            loss_cps_list.append(L_u.item())
        
        writer.add_scalar('lr', get_lr(optimizer), epoch_num)
        writer.add_scalar('cps_w', cps_w, epoch_num)
        writer.add_scalar('loss/total', np.mean(loss_list), epoch_num)
        writer.add_scalar('loss/supervised', np.mean(loss_sup_list), epoch_num)
        writer.add_scalar('loss/consistency', np.mean(loss_cps_list), epoch_num)
        logging.info(f'epoch {epoch_num} : loss : {np.mean(loss_list):.4f}, cps_w: {cps_w:.4f}, lr: {get_lr(optimizer):.6f}')
        
        optimizer.param_groups[0]['lr'] = poly_lr(epoch_num, args.max_epoch, args.base_lr, 0.9)
        
        # === 2. 验证循环 (已更新为 MONAI 滑窗推理) ===
        if (epoch_num > 0 and epoch_num <150 and epoch_num % 10 == 0) or (epoch_num >= 150 and epoch_num % 1 == 0):
        # if epoch_num >= 0 and epoch_num % 1 == 0:
            dice_list_for_epoch = [[] for _ in range(config.num_cls - 1)]
            model.eval()

            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"epoch {epoch_num} validation"):
                    image, gt = fetch_data(batch)
                    pred_indices = sliding_window_inference_skcdf(
                        image=image,
                        model=model,
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
                
                # class_names= ['spleen', 'right kidney', 'left kidney', 'gallbladder', 'esophagus', 'liver', 'stomach', 'aorta', 'inferior vena cava', 'pancreas', 'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate/uterus']
                class_names= ['liver', 'spleen', 'kidney L', 'kidney R', 'stomach', 'gallbladder', 'esophagus',  'pancreas', 'duodenum', 'colon', 'intestine', 'adrenal', 'rectum', 'bladder', 'femur']
                dice_dict = {name: f"{d:.4f}" for name, d in zip(class_names, dice_mean_per_class)}
                logging.info(f'Per-class Dice: {dice_dict}')

                if avg_dice > best_eval:
                    best_eval = avg_dice
                    best_epoch = epoch_num
                    epochs_no_improve = 0
                    save_path = os.path.join(snapshot_path, 'ckpts/best_model.pth')
                    torch.save({'A': model.state_dict(), 'B': model.state_dict()}, save_path)
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