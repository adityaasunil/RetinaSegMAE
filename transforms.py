import albumentations

def get_transforms(split):
        return albumentations.Compose([
            albumentations.Resize(384,384),
            albumentations.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            albumentations.ToTensorV2()
        ])
    