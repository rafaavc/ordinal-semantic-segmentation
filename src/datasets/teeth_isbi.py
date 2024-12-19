import numpy as np, torch
from .dataset import DatasetProvider
from skimage.io import imread
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


NUM_CLASSES = 4
ROOT_PATH = "/data/bioeng/ordinal-segmentation/datasets"
TARGET_SHAPE = (256, 256)


class TeethISBI(torch.utils.data.Dataset):
    K = NUM_CLASSES

    def __init__(self, root, split, transform=None):
        assert split in ('train', 'test')
        root = os.path.join(root, 'teeth-ISBI')
        if split == 'train':
            self.imgs_dir = os.path.join(root, 'v2-TrainingData')
            self.labels_dir = os.path.join(root, 'v2-TrainingData')
        else:
            self.imgs_dir = os.path.join(root, 'Test1Data')
            self.labels_dir = os.path.join(root, 'evaluation_test1', 'manual', 'test1')
        self.images = sorted([f for f in os.listdir(self.imgs_dir) if f.endswith('.bmp')])
        if split == 'train':
            self.labels = [os.path.join(f[:-4], f[:-4]) for f in self.images]
        else:
            self.labels = [f[:-6] for f in sorted(os.listdir(self.labels_dir))]
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = imread(os.path.join(self.imgs_dir, self.images[i]), True)
        masks = [imread(f'{self.labels_dir}/{self.labels[i]}_{j}.bmp', True)
            for j in range(2, 5)]
        masks = [mask > (mask.max()+mask.min())/2 for mask in masks]
        # weirdly, a couple masks have different shape than image -> fix it
        masks = [mask if image.shape == mask.shape else resize(mask, image.shape) for mask in masks]
        if self.split == 'test':
            # train background is black (0), test background is white (255)
            masks = [~mask for mask in masks]
        # fill up holes to since there is some discrepancy between the classes
        for j in range(len(masks)):
            masks[j] = binary_fill_holes(sum(masks[j:]))
        mask = np.sum(masks, 0)
        image = image.astype(np.float32)
        if self.transform:
            d = self.transform(image=image, mask=mask)
            image, mask = d['image'], d['mask']
            mask = mask.long()
        return image, mask


class TeethISBI_DP(DatasetProvider):
    def __init__(self, scale: float, mask_type: str):
        print("WARNING: 'scale' and 'mask_type' parameters aren't being used in TeethISBI dataset!")
        super().__init__(scale, NUM_CLASSES)
        self.transf = A.Compose([
            A.Resize(*TARGET_SHAPE),
            A.Rotate(180),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(0.1, 0.1, p=1),
            A.RandomCrop(*TARGET_SHAPE),
            #A.Normalize(0, 1), no need to normalize, image values already between 0 and 1
            ToTensorV2(),
        ])
        self.test_transf = A.Compose([
            A.Resize(*TARGET_SHAPE),
            ToTensorV2(),
        ])
    
    def get_ordinality_tree(self):
        return {
            0: {
                1: {
                    2: {
                        3: None
                    }
                }
            }
        }

    def create_train(self):
        return TeethISBI(ROOT_PATH, 'train', transform=self.transf)

    def create_val(self):
        return TeethISBI(ROOT_PATH, 'train', transform=self.test_transf)

    def create_test(self):
        return TeethISBI(ROOT_PATH, 'test', transform=self.test_transf)
    
    def get_num_channels(self):
        return 1
