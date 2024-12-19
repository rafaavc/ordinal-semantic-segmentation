import os, albumentations as A
from .dataset import DatasetProvider, SampledDataset
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from .ordinality_tree import OrdinalityTree


NUM_CLASSES = 4
ROOT_PATH = "/data/bioeng/ordinal-segmentation/datasets"
TARGET_SHAPE = (256, 256)


class Iris(SampledDataset):
    K = NUM_CLASSES
    
    def __init__(self, root, split, dataset_scale, transform=None):
        super().__init__()
        assert split in ('train', 'test')
        self.transform = transform

        root = os.path.join(root, 'Iris', 'subset')
        self.imgs_dir = os.path.join(root, 'imgs')
        self.masks_dir = os.path.join(root, 'masks')
        files = [f.removesuffix('.jpg') for f in os.listdir(self.imgs_dir) if f.endswith('.jpg')]

        self.set_files(files, split)
        self.sample_files(dataset_scale, split)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]

        image = imread(os.path.join(self.imgs_dir, file+'.jpg'))
        mask = imread(os.path.join(self.masks_dir, file+'.bmp'), True) / 60

        if self.transform:
            d = self.transform(image=image, mask=mask)
            image, mask = d['image'], d['mask']
            mask = mask.long()
            if len(mask[mask == 3]) == 0 and 4 in mask.unique():
                mask[mask == 4] = 3
        
        image = image.float()
        return image, mask


class Iris_DP(DatasetProvider):
    def __init__(self, scale: float = 1, mask_type: str = None):
        print("WARNING: 'mask_type' parameters aren't being used in Iris dataset!")
        super().__init__(scale, NUM_CLASSES)
        self.transf = A.Compose([
            A.Resize(*TARGET_SHAPE),
            A.Rotate(180),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(0.1, 0.1, p=1),
            A.RandomCrop(*TARGET_SHAPE),
            A.Normalize(0, 1),
            ToTensorV2(),
        ])
        self.test_transf = A.Compose([
            A.Resize(*TARGET_SHAPE),
            A.Normalize(0, 1),
            ToTensorV2(),
        ])
    
    def get_ordinality_tree(self):
        return OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: None
                    }
                }
            }
        })

    def create_train(self):
        return Iris(ROOT_PATH, 'train', dataset_scale=self.scale, transform=self.transf)

    def create_val(self):
        return Iris(ROOT_PATH, 'train', dataset_scale=self.scale, transform=self.test_transf)

    def create_test(self):
        return Iris(ROOT_PATH, 'test', dataset_scale=self.scale, transform=self.test_transf)
    
    def get_num_channels(self):
        return 3
