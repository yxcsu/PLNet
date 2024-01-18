from albumentations import (
    HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, RandomBrightnessContrast, Perspective, CLAHE, 
    Transpose, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, ColorJitter, GaussNoise, MotionBlur, MedianBlur,
    Emboss, Sharpen, Flip, OneOf, SomeOf, Compose, Normalize, CoarseDropout, CenterCrop, GridDropout, Resize,RandomResizedCrop
)
from albumentations.pytorch import ToTensorV2
# set data augment parameters
def data_augment():
    train_transform = Compose([
            OneOf([
            CoarseDropout(p=0.5),
            GaussNoise(p=0.5),
            ], p=0.2),
            OneOf([
            MotionBlur(p=0.5),  
            MedianBlur(blur_limit=3, p=0.5),  
            Blur(blur_limit=3, p=0.5),  
            ], p=0.2),
            SomeOf([
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ], n=3, p=0.6),
            RandomResizedCrop(224, 224, scale=(0.25, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
            Normalize(mean=[0.5762883,0.45526023,0.32699665], std=[0.08670782,0.09286641,0.09925108], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    
    test_transform = Compose([
            CenterCrop(224, 224),
            Normalize(mean=[0.5762883,0.45526023,0.32699665], std=[0.08670782,0.09286641,0.09925108], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)       
    return train_transform,test_transform