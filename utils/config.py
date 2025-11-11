
class Config:
    def __init__(self,task):
        if task == "word":
            self.base_dir = '/data/wangbo/CissMOS/Datasets/Word_merge_npy'
            self.split_dir = {'train':{'image':'imagesTr','label':'labelsTr'},
                    'valid':{'image':'imagesVal','label':'labelsVal'},
                    'test':{'image':'imagesTs','label':'labelsTs'}}
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
        elif task == "flare22":
            self.base_dir = './Datasets/FLARE2022'
            self.save_dir = './flare22_data'
            self.save_dir_nii = '/data/wangbo/SSL_imbalance/flare22_data_nii'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50
        elif task == "amos": # amos
            self.base_dir = '/data/wangbo/CissMOS/Datasets/AMOS22_1.5_2.0_npy'
            self.split_dir = {'train':{'image':'imagesTr','label':'labelsTr'},
                              'valid':{'image':'imagesTr','label':'labelsTr'},
                              'test':{'image':'imagesVa','label':'labelsVa'}}
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50