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
        self.testing_images = os.path.join(PROJECT_ROOT, 'images', 'test')

        if split == "train":
            self.root = os.path.join(PROJECT_ROOT, "images", "train")
        else:
            self.root = os.path.join(PROJECT_ROOT, "images", "test")

        # Take all png files and sort them
        self.files = sorted(
            [f for f in os.listdir(self.root) if f.lower().endswith(".png")]
        )

        self.transforms = get_transforms(split)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.root, fname)

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed for path: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        return img