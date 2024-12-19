import albumentations as A, os, numpy as np, random
from albumentations.pytorch import ToTensorV2
from .dataset import DatasetProvider, SampledDataset
from skimage.io import imread
from datasets.ordinality_tree import OrdinalityTree

TARGET_SHAPE = (256, 256)
ROOT = '/data/auto/bdd100k'


class BDD100K(SampledDataset):
    def __init__(self, files: list[str], img_path: str, tasks: list[str], create_mask, split: str, dataset_scale: float, transform=None):
        super().__init__()
        assert split in ('train', 'test')
        assert img_path in ('100k', '10k')
        self.transform = transform
        self.create_mask = create_mask

        self.set_files(files, split)
        self.sample_files(dataset_scale, split)

        # dataset paths
        self.mask_paths = [ os.path.join(ROOT, "labels", task, "masks") \
                           for task in tasks ]
        self.image_path = os.path.join(ROOT, "images", img_path)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_name = self.files[index]
        target_masks = [
            imread(os.path.join(path, file_name+".png"), True).astype(np.uint8) \
                for path in self.mask_paths
        ]

        mask = self.create_mask(target_masks)

        image = imread(os.path.join(self.image_path, file_name+".jpg"))
        if self.transform:
            d = self.transform(image=image, mask=mask)
            image, mask = d['image'], d['mask']
            mask = mask.long()

        image = image.float()
        return image, mask


class BDD100K_GENERIC_DP(DatasetProvider):
    def __init__(self, scale: float, K: int, dataset, ds_parameters):
        super().__init__(scale, K)
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
        self.dataset = dataset
        self.ds_parameters = ds_parameters

    def handle_mask_type(self, mask_type):
        assert mask_type in mask_types.keys(), f"Unknown mask type '{mask_type}'"
        self.tree = mask_types[mask_type][-1]
        # print("handle_mask_type", type(self.tree))
        return mask_types[mask_type][:-1]
    
    def get_ordinality_tree(self):
        # print("get_ordinality_tree", type(self.tree))
        return self.tree

    def create_train(self):
        return self.dataset(*self.ds_parameters, 'train', self.scale, transform=self.transf)
    
    def create_val(self): # val split is done via dataloader
        return self.dataset(*self.ds_parameters, 'train', self.scale, transform=self.test_transf)

    def create_test(self):
        return self.dataset(*self.ds_parameters, 'test', self.scale, transform=self.test_transf)
    
    def get_num_channels(self):
        return 3


"""
    11: person
    12: rider
    13: car
    14: truck
    15: bus
    16: train
    17: motorcycle
    18: bicycle
"""

def reduced1(target_masks):
    sem_seg, drivable = target_masks

    mask = np.zeros_like(sem_seg) # unknown
    mask[sem_seg != 255] = 1 # environment
    mask[sem_seg == 0] = 2 # road
    mask[sem_seg == 1] = 3 # sidewalk
    mask[(sem_seg >= 11) & (sem_seg <= 18)] = 4 # road agents
    mask[drivable == 1] = 5 # drivable area
    mask[drivable == 0] = 6 # ego lane

    return mask

def reduced1_noagents(target_masks):
    sem_seg, drivable = target_masks

    mask = np.zeros_like(sem_seg) # unknown
    mask[sem_seg != 255] = 1 # environment
    mask[sem_seg == 0] = 2 # road
    mask[sem_seg == 1] = 3 # sidewalk
    mask[drivable == 1] = 4 # drivable area
    mask[drivable == 0] = 5 # ego lane

    return mask

def totalvar_test1(target_masks):
    sem_seg, drivable = target_masks

    mask = np.zeros_like(sem_seg) # unknown
    mask[sem_seg != 255] = 1 # environment
    mask[sem_seg == 0] = 2 # road
    mask[sem_seg == 1] = 3 # sidewalk
    mask[drivable == 0] = 4 # ego lane

    return mask

def wroadagents1_nodrivable(target_masks):
    sem_seg = target_masks[0]

    mask = np.zeros_like(sem_seg) # unknown
    mask[sem_seg != 255] = 1 # environment
    mask[sem_seg == 0] = 2 # road
    mask[sem_seg == 1] = 3 # sidewalk

    # 4, 5, 8, 11 are blank for ordinal segmentation purposes
    # 4 is for "road agents"
    # 5 is for "human"
    mask[sem_seg == 11] = 6 # person
    mask[sem_seg == 12] = 7 # rider
    # 8 is for "two wheels"
    mask[sem_seg == 17] = 9 # motorcycle
    mask[sem_seg == 18] = 10 # bicycle
    # 11 is for "others"
    mask[sem_seg == 13] = 12 # car
    mask[sem_seg == 14] = 13 # truck
    mask[sem_seg == 15] = 14 # bus
    mask[sem_seg == 16] = 15 # train

    return mask

def wroadagents1(target_masks):
    _, drivable = target_masks

    mask = wroadagents1_nodrivable(target_masks)
        
    mask[drivable == 1] = 16 # drivable area
    mask[drivable == 0] = 17 # ego lane

    return mask

def wroadagents1_nodrivable_noabstract(target_masks):
    sem_seg = target_masks[0]

    mask = np.zeros_like(sem_seg) # unknown
    mask[sem_seg != 255] = 1 # environment
    mask[sem_seg == 0] = 2 # road
    mask[sem_seg == 1] = 3 # sidewalk

    mask[sem_seg == 11] = 4 # person
    mask[sem_seg == 12] = 5 # rider

    mask[sem_seg == 17] = 6 # motorcycle
    mask[sem_seg == 18] = 7 # bicycle

    mask[sem_seg == 13] = 8 # car
    mask[sem_seg == 14] = 9 # truck
    mask[sem_seg == 15] = 10 # bus
    mask[sem_seg == 16] = 11 # train

    return mask

def wroadagents1_noabstract(target_masks):
    _, drivable = target_masks

    mask = wroadagents1_nodrivable_noabstract(target_masks)
        
    mask[drivable == 1] = 12 # drivable area
    mask[drivable == 0] = 13 # ego lane

    return mask

def road_only(target_masks):
    sem_seg = target_masks[0]

    mask = np.zeros_like(sem_seg) # unknown
    mask[sem_seg == 0] = 1 # road
    return mask

"""
-> unknown (0)
    -> environment (1)
        -> road (2)
            -> sidewalk (3)
            -> road agents (4) [? this is questionable, agents may not be on the road (even in this more extense definition of road)
                -> human (5)
                    -> person (6)
                    -> rider (7)
                -> two-wheel (8)
                    -> motorcycle (9)
                    -> bicycle (10)
                -> other (11)
                    -> car (12)
                    -> truck (13)
                    -> bus (14)
                    -> train (15)
            -> drivable area (16)
                -> ego lane (17)
"""
mask_types = {
    'road_only': [
        [ "sem_seg" ],
        road_only,
        2,
        None
    ],
    'reduced1': [
        [ "sem_seg", "drivable" ],
        reduced1,
        7,
        OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: None,
                        4: None,
                        5: {
                            6: None
                        }
                    }
                }
            }
        })
    ],
    'reduced1_noagents': [
        [ "sem_seg", "drivable" ],
        reduced1_noagents,
        6,
        OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: None,
                        4: {
                            5: None
                        }
                    }
                }
            }
        })
    ],
    'totalvar_test1': [
        [ "sem_seg", "drivable" ],
        totalvar_test1,
        5,
        OrdinalityTree({
            0: {
                1: {
                    2: {
                        3: None,
                        4: None
                    }
                }
            }
        })
    ],
    'wroadagents1': [
        [ "sem_seg", "drivable" ],
        wroadagents1,
        18,
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
                        },
                        16: {
                            17: None
                        }
                    }
                }
            }
        })
    ],
    'wroadagents1_nodrivable': [
        [ "sem_seg" ],
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
    'wroadagents1_noabstract': [
        [ "sem_seg", "drivable" ],
        wroadagents1_noabstract,
        14,
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
                        11: None,
                        12: {
                            13: None
                        }
                    }
                }
            }
        })
    ],
    'wroadagents1_nodrivable_noabstract': [
        [ "sem_seg" ],
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
    ],
}

def get_image_paths(ds):
    train = [ os.path.join("train", img) for img in os.listdir(os.path.join(ROOT, "images", ds, "train")) ]
    val = [ os.path.join("val", img) for img in os.listdir(os.path.join(ROOT, "images", ds, "val")) ]
    return train + val

class BDDIntersected_DP(BDD100K_GENERIC_DP):
    def __init__(self, scale: float, mask_type: str):
        tasks, create_mask, K = self.handle_mask_type(mask_type)

        bdd10k, bdd100k = [ set(get_image_paths(ds)) for ds in [ "10k", "100k" ] ]
        assert len(bdd10k) == 8000 and len(bdd100k) == 80000

        files = [ f.split(".")[0] for f in bdd10k & bdd100k if f.endswith(".jpg") ]

        super().__init__(scale, K, BDD100K, [ files, "10k", tasks, create_mask ])


class BDD10K_DP(BDD100K_GENERIC_DP):
    def __init__(self, scale: float, mask_type: str):
        self.mask_type = mask_type
        tasks, create_mask, K = self.handle_mask_type(mask_type)

        bdd10k = get_image_paths("10k")
        assert len(bdd10k) == 8000
        files = [ f.split(".")[0] for f in bdd10k if f.endswith(".jpg") ]

        super().__init__(scale, K, BDD100K, [ files, "10k", tasks, create_mask ])
    
    def create_unsupervised(self):
        return BDDUnsupervised_DP(self.scale, self.mask_type).create_train()


class BDDUnsupervised(SampledDataset):
    def __init__(self, bdd100k: list[str], create_mask, split: str, dataset_scale: float, transform=None):
        super().__init__()
        assert split in ('train', 'test')
        self.transform = transform
        self.create_mask = create_mask
        self.image_path = os.path.join(ROOT, "images", "100k")
        self.files = sorted(bdd100k)

        self.sample_files(dataset_scale, split)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        file_name_image = self.files[index]

        image = imread(os.path.join(self.image_path, file_name_image+".jpg"))
        if self.transform:
            d = self.transform(image=image)
            image = d['image']

        image = image.float()
        return image


class BDDUnsupervised_DP(BDD100K_GENERIC_DP):
    def __init__(self, scale: float, mask_type: str):
        tasks, create_mask, K = self.handle_mask_type(mask_type)
        assert len(tasks) == 1 and tasks[0] == "sem_seg"

        random_seeded = random.Random(1)

        bdd10k, bdd100k = [ set(get_image_paths(ds)) for ds in [ "10k", "100k" ] ]
        assert len(bdd10k) == 8000 and len(bdd100k) == 80000

        bdd100k -= bdd10k # remove repeated images

        bdd100k = random_seeded.sample(bdd100k, len(bdd10k) // 6)
        bdd100k = [ f.split(".")[0] for f in bdd100k if f.endswith(".jpg") ]

        print(f"- BDDUnsupervised total files: {len(bdd100k)}")

        super().__init__(scale, K, BDDUnsupervised, [ bdd100k, create_mask ])
    
    def create_test(self):
        raise Exception("BDDUnsupervised doesn't support test dataset.")
    
    def create_val(self):
        raise Exception("BDDUnsupervised doesn't support val dataset.")
