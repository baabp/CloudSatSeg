import albumentations as albu

def training_augmentation_0(resize=(320, 640)):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5)
    ]
    if resize is not None:
        train_transform.append(albu.Resize(resize[0], resize[1]))
    return albu.Compose(train_transform)


def validation_augmentation_0(resize=(320, 640)):
    """Add paddings to make image shape divisible by 32"""
    if resize is not None:
        test_transform = [
            albu.Resize(resize[0], resize[1])
        ]
        return albu.Compose(test_transform)
    else:
        return None