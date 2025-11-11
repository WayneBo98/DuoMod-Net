import torch
import numpy as np

# ====================================================================================
# 核心修改部分：适配非立方体尺寸，并大幅提升效率
# ====================================================================================

def get_part_and_rec_ind_non_cubic(tensor_shape, block_num_per_dim, chnls):
    """
    【已修改，更高效】
    为非立方体数据生成用于“跨图像块重组与恢复”的索引。
    这个版本只生成块级别的索引，非常小且高效，避免了原版生成巨大像素索引导致的显存问题。

    Args:
        tensor_shape (tuple): 输入张量的形状, e.g., (B, C, 64, 128, 128)。
        block_num_per_dim (tuple): 各个维度上的块数量, e.g., (4, 4, 4)。
        chnls (int): 特征图的通道数。

    Returns:
        part_ind (Tensor): 用于打乱块的索引。
        rec_ind (Tensor): 用于恢复块的索引。
    """
    bs, _, d, h, w = tensor_shape
    nd, nh, nw = block_num_per_dim
    cd, ch, cw = d // nd, h // nh, w // nw # 每个块的维度
    
    total_blocks = bs * nd * nh * nw
    
    # 生成一个全局的随机排列，用于打乱所有图像的所有块
    part_ind_flat = torch.randperm(total_blocks, device='cuda')
    # 生成恢复索引
    rec_ind_flat = torch.argsort(part_ind_flat)
    
    # 为了能在 torch.gather 中使用，需要扩展索引的维度以匹配块张量的形状
    # 形状需要是 [total_blocks, chnls, cd, ch, cw]
    # gather 是在第0维操作，所以只需要 [total_blocks, 1, 1, 1, 1] 并让PyTorch自动广播
    part_ind = part_ind_flat.view(-1, 1, 1, 1, 1).expand(-1, chnls, cd, ch, cw)
    rec_ind = rec_ind_flat.view(-1, 1, 1, 1, 1).expand(-1, chnls, cd, ch, cw)
    
    return part_ind, rec_ind

def partition_and_mix(tensor, block_num_per_dim, part_ind):
    """
    【新增】
    一个高效的函数，用于执行实际的“分割与混合”操作。
    """
    bs, c, d, h, w = tensor.shape
    nd, nh, nw = block_num_per_dim
    cd, ch, cw = d // nd, h // nh, w // nw
    
    # 1. 将张量重塑为块的集合: [B, C, D, H, W] -> [B*N_blocks, C, Cd, Ch, Cw]
    blocks = tensor.view(bs, c, nd, cd, nh, ch, nw, cw).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    blocks = blocks.view(bs * nd * nh * nw, c, cd, ch, cw)
    
    # 2. 使用索引打乱块
    mixed_blocks = torch.gather(blocks, dim=0, index=part_ind)
    
    # 3. 将打乱的块重组回张量: [B*N_blocks, ...] -> [B, C, D, H, W]
    mixed_tensor = mixed_blocks.view(bs, nd, nh, nw, c, cd, ch, cw).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    mixed_tensor = mixed_tensor.view(bs, c, d, h, w)
    
    return mixed_tensor

def recover_tensor(mixed_tensor, block_num_per_dim, rec_ind):
    """
    【新增】
    一个高效的函数，用于执行“恢复”操作。
    """
    # 恢复操作的逻辑与混合完全相同，只是使用恢复索引 rec_ind
    return partition_and_mix(mixed_tensor, block_num_per_dim, rec_ind)


def get_loc_mask_non_cubic(patch_shape, block_num_per_dim):
    """
    【已修改】
    为非立方体网格生成位置掩码或ID。

    Args:
        patch_shape (tuple): 整个patch的形状, e.g., (B, C, 64, 128, 128)。
        block_num_per_dim (tuple): 各个维度上的块数量, e.g., (4, 4, 4)。

    Returns:
        loc_list (list): 一个包含每个块位置ID张量的列表。
    """
    nd, nh, nw = block_num_per_dim
    
    loc_list = []
    # 使用 0-indexed 循环，更符合Python习惯
    for z in range(nd):
        for y in range(nh):
            for x in range(nw):
                # 正确的非立方体ID计算公式
                # (z,y,x) -> z * (nh*nw) + y*nw + x
                loc_value = torch.tensor(z * (nh * nw) + y * nw + x)
                loc_list.append(loc_value.view(1, 1)) # 保存为 [1, 1] 的张量以匹配原代码格式

    return loc_list


# ====================================================================================
# OrganClassLogger: 逻辑不变，但用更高效的方式实现
# ====================================================================================

class OrganClassLogger:
    def __init__(self, num_classes=16): # 根据您的任务修改num_classes
        self.num_classes = num_classes
        self.class_total_pixel_store = []
        self.class_dist = torch.ones(num_classes, device='cuda').float() / num_classes

    def append_class_list(self, class_pixels_tensor):
        """ 接收一个扁平化的类别张量 (e.g., shape [N]) """
        self.class_total_pixel_store.append(class_pixels_tensor.detach())

    def update_class_dist(self):
        """
        【已修改，更高效】
        使用 torch.bincount 高效计算类别分布，替代了原来的循环。
        """
        if not self.class_total_pixel_store:
            return
            
        all_pixels = torch.cat(self.class_total_pixel_store, dim=0)
        
        # 使用 bincount 高效计算每个类别的像素数
        counts = torch.bincount(all_pixels, minlength=self.num_classes).float()
        
        # 防止除以零
        if counts.sum() > 0:
            new_dist = counts / counts.sum()
            # 使用EMA（指数移动平均）平滑地更新分布
            self.class_dist = 0.99 * self.class_dist + 0.01 * new_dist.to(self.class_dist.device)
        
        # 清空缓存
        self.class_total_pixel_store = []

    def get_class_dist(self):
        return self.class_dist


# ====================================================================================
# 其他函数：这些函数在我们的主训练流程中没有用到，但为了完整性，也进行了适配
# ====================================================================================

def get_one_block_swap_ind_non_cubic(volume_shape, block_shape):
    """
    【已修改】
    生成用于在batch内图像间交换一个随机非立方体块的索引。
    """
    bs, c, d, h, w = volume_shape
    bd, bh, bw = block_shape

    assert bd <= d and bh <= h and bw <= w, "Block is larger than volume"

    # 1. 创建一个基础的身份索引 (每个像素的索引指向自己)
    base_indices = torch.arange(bs).view(bs, 1, 1, 1, 1).expand(bs, c, d, h, w).cuda()
    
    # 2. 随机选择一个块的起始位置
    d_start = torch.randint(0, d - bd + 1, (1,)).item()
    h_start = torch.randint(0, h - bh + 1, (1,)).item()
    w_start = torch.randint(0, w - bw + 1, (1,)).item()

    # 3. 在这个块的区域内，进行跨图像的随机打乱
    block_region = base_indices[:, :, d_start:d_start+bd, h_start:h_start+bh, w_start:w_start+bw]
    
    # 生成一个随机排列来打乱batch维度
    batch_perm = torch.randperm(bs).cuda()
    swapped_block_region = block_region[batch_perm]

    # 4. 将打乱后的块放回原始索引中
    final_indices = base_indices.clone()
    final_indices[:, :, d_start:d_start+bd, h_start:h_start+bh, w_start:w_start+bw] = swapped_block_region

    return final_indices