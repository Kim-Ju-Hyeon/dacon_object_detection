
import itertools
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_validation_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_test_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(),
    ])

def grid_search_train_transforms(augmentation_config, img_size=512):
    all_augmentations = []
    for aug_name, params in augmentation_config.items():
        AugClass = getattr(A, aug_name)
        augmentation = AugClass(**params)
        all_augmentations.append([None, augmentation])  # Either no augmentation or the augmentation

    for augmentations in itertools.product(*all_augmentations):
        augmentations = [aug for aug in augmentations if aug is not None]  # Remove None values
        aug_list = [str(aug).split('(')[0] for aug in augmentations if aug is not None]
        augmentations.append(A.Resize(img_size, img_size))
        augmentations.append(A.Normalize())
        augmentations.append(ToTensorV2())

        yield A.Compose(augmentations, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])), aug_list