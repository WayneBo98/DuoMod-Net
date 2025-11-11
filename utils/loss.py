import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import normal
import time
import math
softmax_helper = lambda x: F.softmax(x, 1)


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


def cal_class_num(label):
    num_cls = 17
    label_numpy = label.cpu().numpy()
    num_each_class = np.zeros(num_cls)
    for i in range(label_numpy.shape[0]):
        label = label_numpy[i].reshape(-1)
        tmp, _ = np.histogram(label, range(num_cls+1))
        num_each_class += tmp
    return num_each_class.astype(np.float32)

def cal_addtion(cls_num_list,pred,sigma=4):
    cls_list = torch.cuda.FloatTensor(cls_num_list)
    frequency_list = torch.log(cls_list)
    f_list = torch.log(sum(cls_list)) - frequency_list
    sampler = normal.Normal(0, sigma)
    viariation = sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)
    return (viariation.abs().permute(0, 2, 3, 4, 1) / f_list.max() * f_list).permute(0, 4, 1, 2, 3)

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

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight)
    
    def forward(self, input, target):
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

    def update_weight(self, weight):
        self.weight = weight

class DC_and_CE_loss(nn.Module):
    def __init__(self, w_dc=None, w_ce=None, aggregate="sum", weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()

        ce_kwargs = {'weight': w_ce}
        if ignore_label is not None:
            ce_kwargs['reduction'] = 'none'
        
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label
        self.dc = SoftDiceLoss(weight=w_dc, apply_nonlin=softmax_helper)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

    def update_weight(self, weight):
        self.dc.weight = weight
        self.ce.weight = weight

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
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
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class Block_DiceLoss(nn.Module):
    def __init__(self, n_classes, block_num):
        super(Block_DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.block_num = block_num
        self.dice_loss = DiceLoss(self.n_classes)
    def forward(self, inputs, target, weight=None, softmax=False):
        shape = inputs.shape
        img_size = shape[-1]
        div_num = self.block_num
        block_size = math.ceil(img_size / self.block_num)
        if target is not None:
            loss = []
            for i in range(div_num):
                for j in range(div_num):
                    block_features = inputs[:, :, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    block_labels = target[:, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    tmp_loss = self.dice_loss(block_features, block_labels.unsqueeze(1))
                    loss.append(tmp_loss)
            loss = torch.stack(loss).mean()
        return loss


class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight, reduction='none')

    def forward(self, input, target, weight_map=None):
        '''
        - input: B, C, [WHD]
        - target: B, [WHD] / B, 1, [WHD]
        '''
        b = input.shape[0]

        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        # print("\n",input.size(), target.size())
        loss = super().forward(input, target.long()) # B, [WHD]
        loss = loss.view(b, -1)

        if weight_map is not None:
            weight = weight_map.view(b, -1).detach()
            loss = loss * weight
        return torch.mean(loss)

    def update_weight(self, weight):
        self.weight = weight


class ClassDependent_WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, reduction='none'):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight)
        self.reduction = reduction

    def forward(self, input, target, weight_map=None):
        '''
        - input: B, C, [WHD]
        - target: B, [WHD] / B, 1, [WHD]
        '''
        b, c = input.shape[0], input.shape[1]

        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        if weight_map is not None:
            loss = super().forward(input*weight_map.detach(), target.long()) # B, [WHD]
        else:
            loss = super().forward(input, target.long())

        loss = loss.view(b,  -1)

        return torch.mean(loss)

    def update_weight(self, weight):
        self.weight = weight



class Onehot_WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight, reduction='none')

    def forward(self, input, target, weight_map=None):
        '''
        - input: B, C, [WHD]
        - target: B, [WHD] / B, 1, [WHD]
        '''
        b = input.shape[0]

        # print("\n",input.size(), target.size())
        loss = super().forward(input, target) # B, [WHD]
        loss = loss.view(b, -1)

        if weight_map is not None:
            weight = weight_map.view(b, -1).detach()
            loss = loss * weight
        return torch.mean(loss)

    def update_weight(self, weight):
        self.weight = weight
