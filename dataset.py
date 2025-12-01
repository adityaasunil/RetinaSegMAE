from transforms import get_transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import os,sys
import cv2
import matplotlib.pyplot as plt
import torch
from scipy.signal import wiener

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0,PROJECT_ROOT)

class RetinaDataset(Dataset):
    def __init__(self, split):
        self.split = split 

        self.training_images = os.path.join(PROJECT_ROOT, 'images', 'train/image')
        self.training_mask_images = os.path.join(PROJECT_ROOT, 'images', 'train/mask')
        self.testing_images = os.path.join(PROJECT_ROOT, 'images', 'test/image')
        self.testing_mask_images = os.path.join(PROJECT_ROOT, 'images', 'test/mask')

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
        return min(len(self.image_files), len(self.mask_files))
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        img_path = os.path.join(self.root_image, img_name)
        mask_path = os.path.join(self.root_mask, mask_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        green_img = img[:,:,1]

        filtered_green_img = wiener(green_img.astype(np.float32), (5,5))
        filtered_norm = cv2.normalize(filtered_green_img, None, 0, 255, cv2.NORM_MINMAX)
        filtered_normu8 = filtered_norm.astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        enhanced = clahe.apply(filtered_normu8)

        enhanced_f = enhanced.astype(np.float32) / 255

        blurred = cv2.GaussianBlur(enhanced_f, (15,15), 0)
        
        a = 0.3

        vessel_like = enhanced_f - (a*blurred)
        vessel_like = np.clip(vessel_like, 0, None)
        vessel_sharp = 1 / (1 + np.exp(-10*(vessel_like - 0.5)))

        vessel_map = vessel_sharp - vessel_sharp.min()
        vessel_map = vessel_map / vessel_map.max()

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (384,384), interpolation=cv2.INTER_NEAREST)
        mask = mask/255.0
        mask = torch.tensor(mask)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() ==3 and mask.shape[0] == 1:
            pass 
        elif mask.dim() == 3 and mask.shape[0] > 1:
            mask = mask[0].unsqueeze(0)

        transformed = self.transforms(image=img, mask=vessel_map)
        img = transformed['image']
        vessel_mask = transformed['mask']

        if vessel_mask.ndim == 2:
            vessel_mask = vessel_mask.unsqueeze(0)

        combined = torch.cat([img, vessel_mask], dim=0)

        return combined, mask
        

if __name__ == '__main__':
    ds = RetinaDataset('test')
    i,m = ds[19]
    rgb = i[:3].permute(1,2,0).numpy()
    rgb = rgb * 0.5 + 0.5
    vessel = i[3].numpy()

    plt.subplot(1,3,1);plt.title('RGB');plt.imshow(rgb);plt.axis('off')
    plt.subplot(1,3,2);plt.title('Vessel mask');plt.imshow(vessel, cmap='gray');plt.axis('off')
    plt.subplot(1,3,3);plt.title('Ground truth mask');plt.imshow(m.squeeze().numpy(), cmap='gray');plt.axis('off')
    plt.show()