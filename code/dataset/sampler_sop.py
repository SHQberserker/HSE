import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
from tqdm import *

class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size,loss,images_per_class=3):
        self.data_source = data_source
        self.ys = np.array(data_source.sid)
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.ys)
        self.num_classes = len(set(self.ys))
        self.loss=loss


        self.ids = np.array(data_source.ys)
    def __len__(self):
        return self.num_samples

    def __iter__(self):
        print(self.data_source)
        num_batches = len(self.data_source) // self.batch_size
        ret = []

        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=False)
            if(self.loss!='Proxy_Anchor'):
                for i in range(len(sampled_classes)):
                    ith_class_idxs = np.nonzero(self.ys == sampled_classes[i])[0]

                    # selected_idxs = ith_class_idxs[ith_class_idxs]
                    random_idxs = np.random.choice(self.ids[ith_class_idxs], size=2, replace=False)
                
                    fin_selected_id=np.nonzero(self.ids == random_idxs[0])[0]
                    class_sel = np.random.choice(fin_selected_id, size=self.num_instances//2, replace=True)
                    ret.extend(np.random.permutation(class_sel))

                    fin_selected_id=np.nonzero(self.ids == random_idxs[1])[0]
                    class_sel = np.random.choice(fin_selected_id, size=self.num_instances//2, replace=True)
                    ret.extend(np.random.permutation(class_sel))
            else:
                for i in range(len(sampled_classes)):
                    ith_class_idxs = np.nonzero(self.ys == sampled_classes[i])[0]
                    class_sel = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                    ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret) 
        
