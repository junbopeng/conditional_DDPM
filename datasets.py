# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import glob
import random
import os
import numpy as np #V
import torch as torch #V

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/a" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/b" % mode) + "/*.*"))

    def __getitem__(self, index):

        image_A = np.load(self.files_A[index % len(self.files_A)],allow_pickle=True)
        image_B = np.load(self.files_B[index % len(self.files_B)],allow_pickle=True)

        item_A = torch.from_numpy(image_A)
        item_B = torch.from_numpy(image_B)
        
        item_A = torch.unsqueeze(item_A,0)
        item_B = torch.unsqueeze(item_B,0)
        return {"a": item_A, "b": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

