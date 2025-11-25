from transforms import get_transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import os,sys
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0,PROJECT_ROOT)

class RetinaDataset(Dataset):
    def __init__(self, split):
        self.split = split 

        self.training_images = os.path.join(PROJECT_ROOT, 'images', 'train')
        self.training_mask_images = os.path.join(PROJECT_ROOT, 'images', 'train_mask')
        self.testing_images = os.path.join(PROJECT_ROOT, 'images', 'test')
        self.testing_mask_images = os.path.join(PROJECT_ROOT, 'images', 'test_mask')

        if split == "train":
            self.root_image = self.training_images
            self.root_mask = self.training_mask_images
        else:
            self.root_mask = self.testing_mask_images
            self.root_image = self.testing_images

        # Take all png files and sort them
        self.image_files = sorted(
            [f for f in os.listdir(self.root_image) if f.lower().endswith(".png")]
        )
        self.mask_files = sorted(
            [f for f in os.listdir(self.root_mask) if f.lower().endswith('.png')]
        )

        self.transforms = get_transforms(split)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        img_path = os.path.join(self.root_image, img_name)
        mask_path = os.path.join(self.root_mask, mask_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        return img, mask
        

if __name__ == '__main__':
    ds = RetinaDataset('test')
    print(ds[1])