import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import read_list, read_data, softmax

class OrganSeg(Dataset):
    def __init__(self, split='train', transform=None, unlabeled=False, pre_load=False, task="amos", num_cls=1):
        self.ids_list = read_list(split, task=task)
        self.split = split
        self.transform = transform
        self.unlabeled = unlabeled
        self.task = task
        self.num_cls = num_cls
        
        # 使用更明确的参数名 pre_load 代替 is_val 来控制预加载行为
        self.pre_load = pre_load 
        self.data_list = {} # 将 data_list 视为一个缓存

        # 如果需要预加载，则在初始化时填充缓存
        if self.pre_load:
            print("Pre-loading data into memory...")
            for data_id in tqdm(self.ids_list):
                # 调用统一的数据获取方法来填充缓存
                self._get_data(data_id)

    def __len__(self):
        # 这个设计是为了允许在一个 epoch 中重复采样数据
        return len(self.ids_list)

    def _get_data(self, data_id):
        """
        统一的数据获取方法。优先从缓存读取，否则从磁盘读取并存入缓存。
        """
        # 检查缓存
        if data_id in self.data_list:
            return self.data_list[data_id]
        
        # 如果不在缓存中，从磁盘读取
        if self.split =='valid':
            image, label = read_data(data_id, task=self.task, split=self.split)
        elif self.split in ['train','labeled_5p','unlabeled_5p','labeled_10p','unlabeled_10p','labeled_2p','unlabeled_2p','labeled_20p','unlabeled_20p']:
            image, label = read_data(data_id, task=self.task, split='train')
        else:
            image, label = read_data(data_id, task=self.task, split='test')
        
        # 如果开启了预加载，则将首次读取的数据存入缓存
        if self.pre_load:
            self.data_list[data_id] = (image, label)
            
        return image, label


    def __getitem__(self, index):
        # 通过取模运算实现数据的重复采样
        index = index % len(self.ids_list)
        data_id = self.ids_list[index]
        
        # 调用重构后的方法，不再有无用的返回值
        image, label = self._get_data(data_id)
        
        if self.unlabeled: # for safety
            label[:] = 0

        # 预处理步骤
        image = image.clip(min=-125, max=275)
        image = (image + 125) / 400

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_full_image_and_id(self, idx):
        """
        新增方法：获取未经transform的完整图像和ID，用于热力图生成。
        """
        case_id = self.ids_list[idx]
        image, _ = self._get_data(case_id) # 我们只需要图像
        return {'image': image, 'case_id': case_id}

class OrganSegWithID(OrganSeg):
    """
    一个继承自OrganSeg的特殊版本，
    它会在返回的sample字典中额外添加 'case_id'。
    """
    def __getitem__(self, idx):
        data_id = self.ids_list[idx]
        
        # 2. 复用父类的读取方法
        image, label = self._get_data(data_id)

        # 3. 执行父类中的预处理逻辑
        if self.unlabeled:
            label[:] = 0
        image = image.clip(min=-125, max=275)
        image = (image + 125) / 400
        
        # 4. 打包数据，并加入 case_id
        sample = {'image': image, 'label': label, 'case_id': data_id}

        # 5. 应用数据增强 (transform)
        if self.transform:
            sample = self.transform(sample)

        return sample