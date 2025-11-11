import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from tqdm import tqdm
from math import ceil

# 确保可以从你的项目结构中导入 VNet 和 Config
# 例如: from networks.vnet import VNet
# from utils.config import Config
from networks.vnet import VNet
from networks.vnet_skcdf import VNet_Decouple_Attention_ABC
from networks.vnet_dst import VNet_Decoupled # 替换为解耦模型
from networks.vnet_dycon import VNet_dycon
from utils.config import Config

def sliding_window_inference(image, model, patch_size, num_classes, overlap=0.5, device='cuda',exp='dycon'):
    """
    使用滑动窗口对整个3D图像进行推理。
    
    [V2 更新]:
    - 如果图像尺寸小于patch_size，会自动对图像进行零填充（padding）。
    - 改进了循环逻辑，确保图像的边缘和角落能被完整覆盖。
    - 推理结束后，会自动将填充区域裁剪掉，返回与原图大小一致的结果。
    """
    # 1. 获取原始图像尺寸和patch尺寸
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # 2. 计算并应用填充
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # F.pad的填充顺序是 (W_left, W_right, H_top, H_bottom, D_front, D_back)
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

    # 获取填充后图像的尺寸
    padded_D, padded_H, padded_W = image.shape[2:]

    # 3. 初始化输出和计数张量
    # num_classes = model.n_classes
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), device=device)
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), device=device)

    # 4. 计算步长并进行滑动窗口推理
    stride_d = int(patch_d * (1 - overlap)) if patch_d > 1 else 1
    stride_h = int(patch_h * (1 - overlap)) if patch_h > 1 else 1
    stride_w = int(patch_w * (1 - overlap)) if patch_w > 1 else 1
    
    # 确保步长至少为1
    stride_d, stride_h, stride_w = max(1, stride_d), max(1, stride_h), max(1, stride_w)

    # 计算每个维度的滑动次数，确保覆盖到边缘
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        # 确保最后一个窗口的起始位置对齐到图像边缘
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad():
                    if exp =='dycon':
                        _,outputs, _ = model(image_patch)
                    else:
                        outputs = model(image_patch)
                    outputs_softmax = F.softmax(outputs, dim=1)
                
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
    
    # 5. 平均重叠区域的预测
    prediction_map /= (count_map + 1e-8)
    prediction_padded = torch.argmax(prediction_map, dim=1).squeeze(0)

    # 6. 裁剪掉填充区域，恢复到原始尺寸
    final_prediction = prediction_padded[:D, :H, :W]
    
    return final_prediction

def sliding_window_inference_skcdf(image, model, patch_size, num_classes, overlap=0.5, device='cuda'):
    """
    使用滑动窗口对整个3D图像进行推理。
    
    [V2 更新]:
    - 如果图像尺寸小于patch_size，会自动对图像进行零填充（padding）。
    - 改进了循环逻辑，确保图像的边缘和角落能被完整覆盖。
    - 推理结束后，会自动将填充区域裁剪掉，返回与原图大小一致的结果。
    """
    # 1. 获取原始图像尺寸和patch尺寸
    B, C, D, H, W = image.shape
    patch_d, patch_h, patch_w = patch_size

    # 2. 计算并应用填充
    pad_d = max(0, patch_d - D)
    pad_h = max(0, patch_h - H)
    pad_w = max(0, patch_w - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # F.pad的填充顺序是 (W_left, W_right, H_top, H_bottom, D_front, D_back)
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

    # 获取填充后图像的尺寸
    padded_D, padded_H, padded_W = image.shape[2:]

    # 3. 初始化输出和计数张量
    # num_classes = model.n_classes
    prediction_map = torch.zeros((1, num_classes, padded_D, padded_H, padded_W), device=device)
    count_map = torch.zeros((1, 1, padded_D, padded_H, padded_W), device=device)

    # 4. 计算步长并进行滑动窗口推理
    stride_d = int(patch_d * (1 - overlap)) if patch_d > 1 else 1
    stride_h = int(patch_h * (1 - overlap)) if patch_h > 1 else 1
    stride_w = int(patch_w * (1 - overlap)) if patch_w > 1 else 1
    
    # 确保步长至少为1
    stride_d, stride_h, stride_w = max(1, stride_d), max(1, stride_h), max(1, stride_w)

    # 计算每个维度的滑动次数，确保覆盖到边缘
    steps_d = ceil((padded_D - patch_d) / stride_d) + 1 if padded_D > patch_d else 1
    steps_h = ceil((padded_H - patch_h) / stride_h) + 1 if padded_H > patch_h else 1
    steps_w = ceil((padded_W - patch_w) / stride_w) + 1 if padded_W > patch_w else 1

    for i_d in range(steps_d):
        # 确保最后一个窗口的起始位置对齐到图像边缘
        d = min(stride_d * i_d, padded_D - patch_d)
        for i_h in range(steps_h):
            h = min(stride_h * i_h, padded_H - patch_h)
            for i_w in range(steps_w):
                w = min(stride_w * i_w, padded_W - patch_w)
                
                d_end, h_end, w_end = d + patch_d, h + patch_h, w + patch_w
                image_patch = image[:, :, d:d_end, h:h_end, w:w_end]
                
                with torch.no_grad():
                    outputs,_ = model(image_patch, pred_type = "unlabeled")
                    outputs_softmax = F.softmax(outputs, dim=1)
                
                prediction_map[:, :, d:d_end, h:h_end, w:w_end] += outputs_softmax
                count_map[:, :, d:d_end, h:h_end, w:w_end] += 1
    
    # 5. 平均重叠区域的预测
    prediction_map /= (count_map + 1e-8)
    prediction_padded = torch.argmax(prediction_map, dim=1).squeeze(0)

    # 6. 裁剪掉填充区域，恢复到原始尺寸
    final_prediction = prediction_padded[:D, :H, :W]
    
    return final_prediction

class ModelEnsemble(torch.nn.Module):
    def __init__(self, model_A, model_B):
        super(ModelEnsemble, self).__init__()
        self.model_A = model_A
        self.model_B = model_B

    def forward(self, x):
        return (self.model_A(x) + self.model_B(x)) / 2.0

class ModelEnsemble_slc(torch.nn.Module):
    def __init__(self, model_A, model_B):
        super(ModelEnsemble_slc, self).__init__(); self.model_A = model_A; self.model_B = model_B
    def forward(self, x):
        out_A = self.model_A(x); out_soft_A = F.softmax(out_A, dim=1)
        # [忠实实现] 验证时也使用 1 - soft_A 作为输入
        in_B = torch.cat([x, 1 - out_soft_A], dim=1)
        out_B = self.model_B(in_B)
        return (out_soft_A + F.softmax(out_B, dim=1)) / 2.0

def main(args):
    # --- 1. 设置和加载配置 ---
    print("✨ 1. Loading configuration and model...")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = Config(args.task)
    patch_size = config.patch_size
    
    if args.exp =='uamt':
        model = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False  # 推理时关闭 dropout
        ).to(device)

        # b. 加载权重 (关键修改: 加载 ema_model 的权重)
        checkpoint = torch.load(args.model_path)
        # 根据你的训练代码，权重是保存在 'ema_model' 键下的
        model.load_state_dict(checkpoint['ema_model'])  # <-- 核心修改
        model.eval()

        print(f"✅ UAMT Teacher (EMA) model loaded from {args.model_path}")
    elif args.exp =='slcnet':
        model1 = VNet(
        n_channels=config.num_channels, n_classes=config.num_cls, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=False).cuda()
        model2 = VNet(
        n_channels=1 + config.num_cls,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False
        ).cuda()

        # b. 加载权重
        checkpoint = torch.load(args.model_path)
        model1.load_state_dict(checkpoint['A'])
        model2.load_state_dict(checkpoint['B'])
        model1.eval()
        model2.eval()
        model = ModelEnsemble_slc(model1, model2)
    elif args.exp =='skcdf':
        model = VNet_Decouple_Attention_ABC(
        n_channels=config.num_channels, n_classes=config.num_cls, n_filters=config.n_filters,
        normalization='batchnorm', has_dropout=True
    ).cuda()
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['A'])
        model.eval()
    elif args.exp =='dst':
        model_A = VNet_Decoupled(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False
        ).to(device)
        model_B = VNet_Decoupled(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).to(device)

        # b. 加载权重
        checkpoint = torch.load(args.model_path)
        model_A.load_state_dict(checkpoint['A'])
        model_B.load_state_dict(checkpoint['B'])
        model_A.eval()
        model_B.eval()

        # c. 创建用于推理的集成模型
        model = ModelEnsemble(model_A, model_B) # 这里的 model 会被传递给 sliding_window_inference
        print(f"✅ DST ensemble model loaded from {args.model_path}")
    elif args.exp =='dycon':
        model = VNet_dycon(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False
        ).to(device)

        # b. 加载权重
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['ema_model'])
        model.eval()

        # c. 创建用于推理的集成模型
        print(f"✅ Dycon ensemble model loaded from {args.model_path}")
    else:
        # --- 2. 加载模型 ---
        model_A = VNet(
        n_channels=config.num_channels,
        n_classes=config.num_cls,
        n_filters=config.n_filters,
        normalization='batchnorm',
        has_dropout=False
        ).to(device)
        model_B = VNet(
            n_channels=config.num_channels,
            n_classes=config.num_cls,
            n_filters=config.n_filters,
            normalization='batchnorm',
            has_dropout=False
        ).to(device)

        # b. 加载权重
        checkpoint = torch.load(args.model_path)
        model_A.load_state_dict(checkpoint['A'])
        model_B.load_state_dict(checkpoint['B'])
        model_A.eval()
        model_B.eval()

        # c. 创建用于推理的集成模型
        model = ModelEnsemble(model_A, model_B) # 这里的 model 会被传递给 sliding_window_inference
    print(f"✅ SEMI-SUPERVISED ensemble model loaded from {args.model_path}")

    # --- 3. 创建输出目录 ---
    os.makedirs(args.output_path, exist_ok=True)

    # --- 4. 遍历NPY文件并进行推理 ---
    npy_files = [f for f in os.listdir(args.npy_path) if f.endswith('.npy')]
    print(f"🚀 4. Found {len(npy_files)} .npy files. Starting inference...")

    for npy_filename in tqdm(npy_files, desc="Inference Progress"):
        # --- a. 构建文件路径并检查 ---
        nii_filename = npy_filename.replace('_image.npy', '.nii.gz')
        original_nii_filepath = os.path.join(args.original_nii_path, nii_filename)
        npy_filepath = os.path.join(args.npy_path, npy_filename)

        if not os.path.exists(original_nii_filepath):
            print(f"⚠️ Warning: Corresponding file {nii_filename} not found in {args.original_nii_path}. Skipping {npy_filename}.")
            continue

        # --- b. 从.nii.gz文件加载原始元数据 ---
        sitk_image_orig = sitk.ReadImage(original_nii_filepath)
        original_spacing = sitk_image_orig.GetSpacing()
        original_size = sitk_image_orig.GetSize() # (x, y, z)
        original_origin = sitk_image_orig.GetOrigin()
        original_direction = sitk_image_orig.GetDirection()

        # --- c. 从.npy文件加载数据并预处理 ---
        # 1. 加载已经重采样（resampled）的图像数据
        image_np = np.load(npy_filepath).astype(np.float32)

        # 2. 归一化 (假设.npy文件只经过重采样，未进行归一化)
        # 如果你的.npy文件已经归一化，可以注释掉下面这行
        image_np = image_np.clip(min=-125, max=275)
        image_np = (image_np + 125) / 400
        # image_np = (image_np - np.mean(image_np)) / np.std(image_np)

        # 3. 转换为 Tensor
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        image_tensor = image_tensor.to(device)

        if args.exp =='skcdf':
            prediction_tensor = sliding_window_inference_skcdf(
            image=image_tensor,
            model=model,
            patch_size=patch_size,
            num_classes=config.num_cls,
            overlap=args.overlap
        )
        elif args.exp =='dycon':
            prediction_tensor = sliding_window_inference(
                image=image_tensor,
                model=model,
                patch_size=patch_size,
                num_classes=config.num_cls,
                overlap=args.overlap,
                exp = args.exp
            )
        else:
            # --- d. 滑动窗口推理 ---
            prediction_tensor = sliding_window_inference(
                image=image_tensor,
                model=model,
                patch_size=patch_size,
                num_classes=config.num_cls,
                overlap=args.overlap,
                exp = args.exp
            )
        prediction_np = prediction_tensor.cpu().numpy().astype(np.uint8) # (D, H, W)
        # --- e. 后处理：重采样回原始尺寸 ---
        # 预测结果的尺寸与.npy文件的尺寸一致
        # 我们需要将其恢复到原始.nii.gz文件的尺寸
        original_size_np_order = (original_size[2], original_size[1], original_size[0]) # (z, y, x)
        
        resample_back_factor = [
            original_size_np_order[i] / prediction_np.shape[i] for i in range(3)
        ]
        
        # 使用最近邻插值(order=0)来重采样分割掩码
        resampled_prediction_np = zoom(prediction_np, resample_back_factor, order=0, mode='nearest')

        # --- f. 保存为.nii.gz文件，并恢复元数据 ---
        prediction_sitk = sitk.GetImageFromArray(resampled_prediction_np)
        
        prediction_sitk.SetSpacing(original_spacing)
        prediction_sitk.SetOrigin(original_origin)
        prediction_sitk.SetDirection(original_direction)
        
        output_filepath = os.path.join(args.output_path, nii_filename)
        sitk.WriteImage(prediction_sitk, output_filepath)

    print("🎉 All predictions are saved successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D Medical Image Segmentation Inference from .npy files")
    
    # --- 修改了这里的路径参数 ---
    parser.add_argument('--npy_path', type=str, help="Path to the directory containing pre-processed .npy files.", default='/data/wangbo/CissMOS/Datasets/AMOS22_1.5_2.0_npy/imagesTr')
    parser.add_argument('--original_nii_path', type=str, help="Path to the directory with original .nii.gz files for metadata.", default='/data/wangbo/CissMOS/Datasets/Amos22/imagesTr')
    parser.add_argument('--output_path', type=str, help="Path to save the segmentation results.", default='/data/wangbo/CissMOS/training_set_results')
    parser.add_argument('--model_path', type=str, help="Path to the trained model checkpoint (.pth file).", default='/data/wangbo/CissMOS/logs/amos/uncertainty_driven_sampling/disagreement_snapshot/seed_1/ckpts/best_model.pth')
    
    # --- 其他参数保持不变 ---
    parser.add_argument('--exp', type=str, default='uamt', help="GPU ID to use.")
    parser.add_argument('-g', '--gpu', type=str, default='0', help="GPU ID to use.")
    parser.add_argument('--task', type=str, default='amos', help="Task name to load the correct configuration (e.g., 'amos').")
    parser.add_argument('--overlap', type=float, default=0.5, help="Overlap ratio for sliding window, between 0 and 1.")
    
    args = parser.parse_args()
    
    main(args)