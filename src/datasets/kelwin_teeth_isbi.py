import numpy as np, os, albumentations as A
from .dataset import DatasetProvider, SampledDataset
from skimage.io import imread
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize
from albumentations.pytorch import ToTensorV2
from .ordinality_tree import OrdinalityTree


NUM_CLASSES = 5
ROOT_PATH = "/data/bioeng/ordinal-segmentation/datasets"
TARGET_SHAPE = (256, 256)


class KelwinTeethISBI(SampledDataset):
    K = NUM_CLASSES
    
    def __init__(self, root, split, dataset_scale, transform=None):
        super().__init__()
        assert split in ('train', 'test')
        root = os.path.join(root, 'teeth-ISBI')

        self.imgs_dir = os.path.join(root, 'v2-TrainingData')
        self.labels_dir = os.path.join(root, 'v2-TrainingData')

        files = [f for f in os.listdir(self.imgs_dir) if f.endswith('.bmp')]

        self.set_files(files, split)
        self.sample_files(dataset_scale, split)

        self.labels = [os.path.join(f[:-4], f[:-4]) for f in self.files]

        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = imread(os.path.join(self.imgs_dir, self.files[i]), True)
        foreground_name = self.files[i][:-4]
        foreground_name = "0" + foreground_name if len(foreground_name) == 1 else foreground_name 
        masks = [imread(os.path.join(self.labels_dir, "foreground", foreground_name + ".jpg"), True),
                 *[imread(f'{self.labels_dir}/{self.labels[i]}_{j}.bmp', True)
            for j in range(2, 5)]]
        masks = [mask > (mask.max()+mask.min())/2 for mask in masks]
        # weirdly, a couple masks have different shape than image -> fix it
        masks = [mask if image.shape == mask.shape else resize(mask, image.shape) for mask in masks]
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


class KelwinTeethISBI_DP(DatasetProvider):
    def __init__(self, scale: float = 1, mask_type: str = None):
        print("WARNING: 'mask_type' parameters aren't being used in TeethISBI dataset!")
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
        return OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: {
                            4: None
                        }
                    }
                }
            }
        })

    def create_train(self):
        return KelwinTeethISBI(ROOT_PATH, 'train', dataset_scale=self.scale, transform=self.transf)

    def create_val(self):
        return KelwinTeethISBI(ROOT_PATH, 'train', dataset_scale=self.scale, transform=self.test_transf)

    def create_test(self):
        return KelwinTeethISBI(ROOT_PATH, 'test', dataset_scale=self.scale, transform=self.test_transf)
    
    def get_num_channels(self):
        return 1
