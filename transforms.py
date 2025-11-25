import albumentations

def get_transforms(split):
    if split == 'train':
        return albumentations.Compose([
            albumentations.Rotate(p=0.5),
            albumentations.Resize(384,384),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            albumentations.ToTensorV2()
        ])
    else:
        return albumentations.Compose([
            albumentations.Resize(384,384),
            albumentations.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            albumentations.ToTensorV2()
        ])
    