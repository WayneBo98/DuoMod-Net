import os
import math
from tqdm import tqdm
import numpy as np
import random
import SimpleITK as sitk
from math import ceil
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import GradScaler, autocast
from utils.config import Config

def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


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

def calculate_class_distribution(label_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    高效地计算批次中标注图的类别分布，完全在GPU上运行。
    """
    # 确保输入是Long类型，并且没有channel=1的维度
    labels = label_tensor.long()
    if labels.dim() == 5 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
        
    one_hot = F.one_hot(labels, num_classes=num_classes)
    # 沿着所有非类别维度（B, D, H, W）求和
    counts = one_hot.sum(dim=tuple(range(one_hot.dim() - 1)))
    return counts.float()


def print_func(item):
    # print(type(item))
    if type(item) == torch.Tensor:
        return [round(x,4) for x in item.data.cpu().numpy().tolist()]
    elif type(item) == np.ndarray:
        return [round(x,4) for x in item.tolist()]
    else:
        raise TypeError


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=0, keepdims=True)
    s = x_exp / (x_sum)
    return s

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr


def read_list(split, task):
    config = Config(task)
    ids_list = np.loadtxt(
        os.path.join(config.base_dir, 'splits', f'{split}.txt'),
        dtype=str
    ).tolist()
    return sorted(ids_list)


def read_data(data_id, task, split, nifti=False, test=False):
    config = Config(task) 
    im_path = os.path.join(config.base_dir,config.split_dir[split]['image'], f'{data_id}_image.npy')
    lb_path = os.path.join(config.base_dir, config.split_dir[split]['label'],f'{data_id}_label.npy')
    if not os.path.exists(im_path) or not os.path.exists(lb_path):
        raise ValueError(data_id)
    image = np.load(im_path)
    label = np.load(lb_path)

    return image, label


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fetch_data(batch, labeled=True):
    image = batch['image'].cuda()
    if labeled:
        label = batch['label'].cuda().unsqueeze(1)
        return image, label
    else:
        return image

def sliding_window_inference_skcdf(image, model, patch_size, num_classes, overlap=0.5, device='cuda', return_prob=False):
    """
    使用滑动窗口进行推理（V4 OOM终极解决方案）。
    采用“大图驻留GPU，结果返回CPU”的混合策略，平衡速度与显存。
    """
    # 确保模型在GPU上
    model.to(device)
    # 确保输入图像在GPU上 (这是为了高效切片)
    image = image.to(device)
    
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # Padding (在GPU上完成)
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    
    padded_D, padded_H, padded_W = image.shape[2:]

    # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 1: 在CPU上创建累加器 !!!!!!!!!!!!!!!!!!!!!!!!!
    # 使用float16节省内存
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 计算步长并进行滑动窗口推理
    stride_d, stride_h, stride_w = [max(1, int(p * (1 - overlap))) for p in patch_size]
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                
                # a. 在GPU上高效切片
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad(), autocast(device_type=device.type if isinstance(device, torch.device) else device):
                    # b. 在GPU上进行高速计算
                    outputs,_ = model(image_patch, pred_type = "unlabeled")
                    outputs_softmax = F.softmax(outputs, dim=1)
                
                # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 2: 将小结果送回CPU进行累加 !!!!!!!!!!!!!!!!!!!!!!!!!
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax.to(torch.float16)
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 5. 平均、argmax等后续操作都在CPU上完成
    prediction_map /= (count_map + 1e-8)
    
    if return_prob:
        # 返回CPU上的概率图
        return prediction_map[:, :, :D, :H, :W] 
    else:
        prediction_padded = torch.argmax(prediction_map.float(), dim=1)
        final_prediction = prediction_padded[:, :D, :H, :W]
        return final_prediction # 返回一个 [D, H, W] 的CPU Tensor

def sliding_window_inference(image, model, patch_size, num_classes, overlap=0.5, device='cuda', return_prob=False):
    """
    使用滑动窗口进行推理（V4 OOM终极解决方案）。
    采用“大图驻留GPU，结果返回CPU”的混合策略，平衡速度与显存。
    """
    # 确保模型在GPU上
    model.to(device)
    # 确保输入图像在GPU上 (这是为了高效切片)
    image = image.to(device)
    
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # Padding (在GPU上完成)
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    
    padded_D, padded_H, padded_W = image.shape[2:]

    # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 1: 在CPU上创建累加器 !!!!!!!!!!!!!!!!!!!!!!!!!
    # 使用float16节省内存
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 计算步长并进行滑动窗口推理
    stride_d, stride_h, stride_w = [max(1, int(p * (1 - overlap))) for p in patch_size]
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                
                # a. 在GPU上高效切片
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad(), autocast(device_type=device.type if isinstance(device, torch.device) else device):
                    # b. 在GPU上进行高速计算
                    outputs = model(image_patch)
                    outputs_softmax = F.softmax(outputs, dim=1)
                
                # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 2: 将小结果送回CPU进行累加 !!!!!!!!!!!!!!!!!!!!!!!!!
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax.to(torch.float16)
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 5. 平均、argmax等后续操作都在CPU上完成
    prediction_map /= (count_map + 1e-8)
    
    if return_prob:
        # 返回CPU上的概率图
        return prediction_map[:, :, :D, :H, :W] 
    else:
        prediction_padded = torch.argmax(prediction_map.float(), dim=1)
        final_prediction = prediction_padded[:, :D, :H, :W]
        return final_prediction # 返回一个 [D, H, W] 的CPU Tensor

def sliding_window_inference_mutual(image, model, patch_size, num_classes, overlap=0.5, device='cuda', return_prob=False):
    """
    使用滑动窗口进行推理（V4 OOM终极解决方案）。
    采用“大图驻留GPU，结果返回CPU”的混合策略，平衡速度与显存。
    """
    # 确保模型在GPU上
    model.to(device)
    # 确保输入图像在GPU上 (这是为了高效切片)
    image = image.to(device)
    
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # Padding (在GPU上完成)
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    
    padded_D, padded_H, padded_W = image.shape[2:]

    # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 1: 在CPU上创建累加器 !!!!!!!!!!!!!!!!!!!!!!!!!
    # 使用float16节省内存
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 计算步长并进行滑动窗口推理
    stride_d, stride_h, stride_w = [max(1, int(p * (1 - overlap))) for p in patch_size]
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                
                # a. 在GPU上高效切片
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad(), autocast(device_type=device.type if isinstance(device, torch.device) else device):
                    # b. 在GPU上进行高速计算
                    outputs,_ = model(image_patch)
                    outputs_softmax = F.softmax(outputs[0], dim=1)
                
                # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 2: 将小结果送回CPU进行累加 !!!!!!!!!!!!!!!!!!!!!!!!!
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax.to(torch.float16)
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 5. 平均、argmax等后续操作都在CPU上完成
    prediction_map /= (count_map + 1e-8)
    
    if return_prob:
        # 返回CPU上的概率图
        return prediction_map[:, :, :D, :H, :W] 
    else:
        prediction_padded = torch.argmax(prediction_map.float(), dim=1)
        final_prediction = prediction_padded[:, :D, :H, :W]
        return final_prediction # 返回一个 [D, H, W] 的CPU Tensor


def sliding_window_inference_dcnet(image, model, patch_size, num_classes, overlap=0.5, device='cuda', return_prob=False):
    """
    使用滑动窗口进行推理（V4 OOM终极解决方案）。
    采用“大图驻留GPU，结果返回CPU”的混合策略，平衡速度与显存。
    """
    # 确保模型在GPU上
    model.to(device)
    # 确保输入图像在GPU上 (这是为了高效切片)
    image = image.to(device)
    
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # Padding (在GPU上完成)
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    
    padded_D, padded_H, padded_W = image.shape[2:]

    # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 1: 在CPU上创建累加器 !!!!!!!!!!!!!!!!!!!!!!!!!
    # 使用float16节省内存
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 计算步长并进行滑动窗口推理
    stride_d, stride_h, stride_w = [max(1, int(p * (1 - overlap))) for p in patch_size]
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                
                # a. 在GPU上高效切片
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad(), autocast(device_type=device.type if isinstance(device, torch.device) else device):
                    # b. 在GPU上进行高速计算
                    outputs,_,_,_,_ = model(image_patch)
                    outputs_softmax = F.softmax(outputs, dim=1)
                
                # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 2: 将小结果送回CPU进行累加 !!!!!!!!!!!!!!!!!!!!!!!!!
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax.to(torch.float16)
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 5. 平均、argmax等后续操作都在CPU上完成
    prediction_map /= (count_map + 1e-8)
    
    if return_prob:
        # 返回CPU上的概率图
        return prediction_map[:, :, :D, :H, :W] 
    else:
        prediction_padded = torch.argmax(prediction_map.float(), dim=1)
        final_prediction = prediction_padded[:, :D, :H, :W]
        return final_prediction # 返回一个 [D, H, W] 的CPU Tensor

def sliding_window_inference_dycon(image, model, patch_size, num_classes, overlap=0.5, device='cuda', return_prob=False):
    """
    使用滑动窗口进行推理（V4 OOM终极解决方案）。
    采用“大图驻留GPU，结果返回CPU”的混合策略，平衡速度与显存。
    """
    # 确保模型在GPU上
    model.to(device)
    # 确保输入图像在GPU上 (这是为了高效切片)
    image = image.to(device)
    
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # Padding (在GPU上完成)
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    
    padded_D, padded_H, padded_W = image.shape[2:]

    # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 1: 在CPU上创建累加器 !!!!!!!!!!!!!!!!!!!!!!!!!
    # 使用float16节省内存
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), dtype=torch.float16).cuda()
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # 计算步长并进行滑动窗口推理
    stride_d, stride_h, stride_w = [max(1, int(p * (1 - overlap))) for p in patch_size]
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                
                # a. 在GPU上高效切片
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad(), autocast(device_type=device.type if isinstance(device, torch.device) else device):
                    # b. 在GPU上进行高速计算
                    _, outputs, _ = model(image_patch)
                    outputs_softmax = F.softmax(outputs, dim=1)
                
                # !!!!!!!!!!!!!!!!!!!!!!!!! 关键修正 2: 将小结果送回CPU进行累加 !!!!!!!!!!!!!!!!!!!!!!!!!
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax.to(torch.float16)
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 5. 平均、argmax等后续操作都在CPU上完成
    prediction_map /= (count_map + 1e-8)
    
    if return_prob:
        # 返回CPU上的概率图
        return prediction_map[:, :, :D, :H, :W] 
    else:
        prediction_padded = torch.argmax(prediction_map.float(), dim=1)
        final_prediction = prediction_padded[:, :D, :H, :W]
        return final_prediction # 返回一个 [D, H, W] 的CPU Tensor

def test_all_case(net, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case(
            net, 
            image, 
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_all_case_dc(net, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case_dc(
            net, 
            image, 
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_all_case_dycon(net, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case_dycon(
            net, 
            image, 
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    # print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)
    #
    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                y1 = net(test_patch) # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map

def test_single_case_dc(net, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    # print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)
    #
    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                # y1 = net(test_patch) # <--
                output1, output2, encoder_features, decoder_features1, decoder_features2 = net(test_patch)
                y1 = (output1 + output2) / 2.
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map

def test_single_case_dycon(net, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    
    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                _,y1,_ = net(test_patch) # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map

def test_all_case_AB(net_A, net_B, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        if task == "synapse":
            pred, _ = test_single_case_AB(
                net_A, net_B,
                image,
                stride_xy,
                stride_z,
                patch_size,
                num_classes=num_classes
            )
        else:
            pred, _ = test_single_case_AB(
                net_A, net_B,
                image,
                stride_xy,
                stride_z,
                patch_size,
                num_classes=num_classes
            )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_all_case_AB_CDMAD(net_A, net_B, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case_AB_CDMAD(
            net_A, net_B,
            image,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_all_case_AB_SLCNet(net_A, net_B, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        pred, _ = test_single_case_AB_SLCNET(
            net_A, net_B,
            image,
            stride_xy,
            stride_z,
            patch_size,
            num_classes=num_classes
        )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_all_case_AB_fcc(net_A, net_B, ids_list, task, num_classes, patch_size, stride_xy, stride_z, test_save_path=None):
    for data_id in tqdm(ids_list):
        image, _ = read_data(data_id, task, test=True, normalize=True)
        if task == "synapse":
            pred, _ = test_single_case_AB_synapse(
                net_A, net_B,
                image,
                stride_xy,
                stride_z,
                patch_size,
                num_classes=num_classes
            )
        else:
            pred, _ = test_single_case_AB_fcc(
                net_A, net_B,
                image,
                stride_xy,
                stride_z,
                patch_size,
                num_classes=num_classes
            )
        out = sitk.GetImageFromArray(pred.astype(np.float32))
        sitk.WriteImage(out, f'{test_save_path}/{data_id}.nii.gz')

def test_single_case_AB_synapse(net_A, net_B, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    # print(image.shape)

    image = torch.FloatTensor(image).unsqueeze(0)
    image = F.interpolate(image, size=(dd, ww//2, hh//2),mode='trilinear', align_corners=False)
    image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                y1 = (net_A(test_patch) + net_B(test_patch)) / 2.0 # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map



def test_single_case_AB(net_A, net_B, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    # print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)

    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                y1 = (net_A(test_patch) + net_B(test_patch)) / 2.0 # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map

def test_single_case_AB_CDMAD(net_A, net_B, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)

    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch = test_patch.transpose(2, 4)
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                white = torch.ones_like(test_patch)
                # print(white.shape)
                # white = torch.ones(1,1,test_patch.shape[2],test_patch.shape[3],test_patch.shape[4]).cuda()
                pseudo_A = net_A(white)
                pseudo_B = net_B(white)
                pseudo_A = nn.AdaptiveAvgPool3d((1,1,1))(pseudo_A).repeat(test_patch.shape[0],1,test_patch.shape[2],test_patch.shape[3],test_patch.shape[4])
                pseudo_B = nn.AdaptiveAvgPool3d((1,1,1))(pseudo_B).repeat(test_patch.shape[0],1,test_patch.shape[2],test_patch.shape[3],test_patch.shape[4])
                # pseudo_A[:,0,:,:,:] = 0
                # pseudo_B[:,0,:,:,:] = 0
                y1 = (net_A(test_patch)-pseudo_A + net_B(test_patch)-pseudo_B) / 2.0 # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map

def test_single_case_AB_SLCNET(net_A, net_B, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)

    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch = test_patch.transpose(2, 4)
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                output_A = net_A(test_patch)
                outputs_soft_A = torch.softmax(output_A, dim=1)
                y1 = (output_A + net_B(torch.cat([test_patch,1-outputs_soft_A],dim=1))) / 2.0
                # y1 = (net_A(test_patch) + net_B(test_patch)) / 2.0 # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map

def test_single_case_AB_fcc(net_A, net_B, image, stride_xy, stride_z, patch_size, num_classes):
    image = image[np.newaxis]
    _, dd, ww, hh = image.shape
    print(image.shape)
    # resize_shape=(patch_size[0]+patch_size[0]//4,
    #               patch_size[1]+patch_size[1]//4,
    #               patch_size[2]+patch_size[2]//4)

    # image = torch.FloatTensor(image).unsqueeze(0)
    # image = F.interpolate(image, size=resize_shape,mode='trilinear', align_corners=False)
    # image = image.squeeze(0).numpy()

    image = image.transpose(0, 3, 2, 1) # <-- take care the shape
    # print(image.shape)
    patch_size = (patch_size[2], patch_size[1], patch_size[0])
    _, ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + image.shape[1:4]).astype(np.float32)
    cnt = np.zeros(image.shape[1:4]).astype(np.float32)
    # print("score_map", score_map.shape)
    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # print("test", test_patch.shape)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # print("===",test_patch.size())
                # <-- [1, 1, Z, Y, X] => [1, 1, X, Y, Z]
                test_patch = test_patch.transpose(2, 4)
                fake = torch.zeros_like(test_patch).cuda()
                # output_A_ini = net_A(test_patch,fake,-1,mode='test')
                # output_B_ini = net_B(test_patch,fake,-1,mode='test')
                # max_A_ini = torch.argmax(output_A_ini.detach(), dim=1, keepdim=True).long()
                # max_B_ini = torch.argmax(output_B_ini.detach(), dim=1, keepdim=True).long()

                output_A = net_A(test_patch,fake,1,0,'test')
                output_B = net_B(test_patch,fake,1,0,'test')
                y1 = (output_A + output_B) / 2.0 # <--
                y = F.softmax(y1, dim=1) # <--
                y = y.cpu().data.numpy()
                y = y[0, ...]
                y = y.transpose(0, 3, 2, 1)
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    # print("score_map", score_map.shape)
    # print("score_map", cnt.shape)

    score_map = score_map / np.expand_dims(cnt, axis=0) # [Z, Y, X]
    score_map = score_map.transpose(0, 3, 2, 1) # => [X, Y, Z]
    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map
