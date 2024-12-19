import albumentations as A, os, numpy as np, random
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from .dataset import DatasetProvider
from skimage.io import imread
from .ordinality_tree import OrdinalityTree

TARGET_SHAPE = (256, 256)
ROOT = '/data/auto/cityscapes'


class Cityscapes(Dataset):
    def __init__(self, create_mask, split: str, dataset_scale: float, transform=None):
        assert split in ('train', 'test', 'val')
        self.transform = transform
        self.create_mask = create_mask
        random_seeded = random.Random(1)


        self.mask_path = os.path.join(ROOT, "gtFine", split)
        self.image_path = os.path.join(ROOT, "leftImg8bit", split)
        imgs = [
            os.path.join(location, img.removesuffix("_leftImg8bit.png"))
                for location in os.listdir(self.image_path)
                for img in os.listdir(os.path.join(self.image_path, location))
                    if os.path.isdir(os.path.join(self.image_path, location)) and img.endswith("png")
        ]

        # Dataset scale sampling
        original_size = len(imgs)
        imgs = sorted(imgs) # sort for guarantees that it is repeatable
        self.files = random_seeded.sample(imgs, int(dataset_scale * original_size))

        print(f"Dataset size = {dataset_scale} * {original_size} = {len(self.files)}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_name = self.files[index]

        image = imread(os.path.join(self.image_path, file_name+"_leftImg8bit.png"))
        mask = imread(os.path.join(self.mask_path, file_name+"_gtFine_labelIds.png"), True)
        mask = self.create_mask(mask)

        if self.transform:
            d = self.transform(image=image, mask=mask)
            image, mask = d['image'], d['mask']
            mask = mask.long()

        image = image.float()
        return image, mask


class Cityscapes_DP(DatasetProvider):
    def __init__(self, scale: float, mask_type: str):
        assert mask_type in mask_types.keys(), f"Unknown mask type '{mask_type}'"
        self.create_mask, k, self.tree = mask_types[mask_type]
        super().__init__(scale, k)

        self.transf = A.Compose([
            A.Resize(*TARGET_SHAPE),
            # A.Rotate(180),
            # A.HorizontalFlip(),
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
        return self.tree

    def create_train(self):
        return Cityscapes(self.create_mask, 'train', self.scale, transform=self.transf)
    
    def create_val(self):
        return Cityscapes(self.create_mask, 'val', self.scale, transform=self.test_transf)

    def create_test(self):
        return Cityscapes(self.create_mask, 'test', self.scale, transform=self.test_transf)
    
    def get_num_channels(self):
        return 3


def wroadagents1_nodrivable(target_mask):
    mask = np.zeros_like(target_mask) # unknown

    mask[target_mask >= 4] = 1 # environment - starting at static
    mask[(target_mask == 7) | (target_mask == 9) | (target_mask == 10)] = 2 # road - road + parking + rail track
    mask[target_mask == 8] = 3 # sidewalk
    # 4, 5, 8, 11 are blank for ordinal segmentation purposes
    # 4 is for "road agents"
    # 5 is for "human"
    mask[target_mask == 24] = 6 # person
    mask[target_mask == 25] = 7 # rider
    # 8 is for "two wheels"
    mask[target_mask == 32] = 9 # motorcycle
    mask[target_mask == 33] = 10 # bicycle
    # 11 is for "others"
    mask[target_mask == 26] = 12 # car
    mask[(target_mask == 27) | (target_mask == 29) | (target_mask == 30)] = 13 # truck - truck + caravan + trailer
    mask[target_mask == 28] = 14 # bus
    mask[target_mask == 31] = 15 # train

    return mask

def wroadagents1_nodrivable_noabstract(target_mask):
    mask = np.zeros_like(target_mask) # unknown

    mask[target_mask >= 4] = 1 # environment - starting at static
    mask[(target_mask == 7) | (target_mask == 9) | (target_mask == 10)] = 2 # road - road + parking + rail track
    mask[target_mask == 8] = 3 # sidewalk

    mask[target_mask == 24] = 4 # person
    mask[target_mask == 25] = 5 # rider

    mask[target_mask == 32] = 6 # motorcycle
    mask[target_mask == 33] = 7 # bicycle

    mask[target_mask == 26] = 8 # car
    mask[(target_mask == 27) | (target_mask == 29) | (target_mask == 30)] = 9 # truck - truck + caravan + trailer
    mask[target_mask == 28] = 10 # bus
    mask[target_mask == 31] = 11 # train

    return mask


mask_types = {
    'wroadagents1_nodrivable': [
        wroadagents1_nodrivable,
        16,
        OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: None,
                        4: {
                            'is_abstract': None,
                            5:  {
                                'is_abstract': None,
                                6: None,
                                7: None
                            },
                            8: {
                                'is_abstract': None,
                                9: None,
                                10: None
                            },
                            11: {
                                'is_abstract': None,
                                12: None,
                                13: None,
                                14: None,
                                15: None
                            }
                        }
                    }
                }
            }
        })
    ],
    'wroadagents1_nodrivable_noabstract': [
        wroadagents1_nodrivable_noabstract,
        12,
        OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: None,
                        4: None,
                        5: None,
                        6: None,
                        7: None,
                        8: None,
                        9: None,
                        10: None,
                        11: None
                    }
                }
            }
        })
    ]
}
