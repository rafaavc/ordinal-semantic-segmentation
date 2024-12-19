from typing import Optional, Callable, Tuple, Any
from .dataset import DatasetProvider
from torchvision import transforms as tf
from torchvision.datasets import CocoDetection

TARGET_SIZE = (256, 256)

NUM_CLASSES = 21
COCO_ROOT_PATH = "/data/toy/coco"

COCO_TRAIN_PATH = COCO_ROOT_PATH + "/train2017"
COCO_VAL_PATH = COCO_ROOT_PATH + "/val2017"
COCO_TEST_PATH = COCO_ROOT_PATH + "/test2017"

COCO_TRAIN_ANNOTATIONS_PATH = COCO_ROOT_PATH + "/annotations/instances_train2017.json"
COCO_VAL_ANNOTATIONS_PATH = COCO_ROOT_PATH + "/annotations/instances_val2017.json"


class CocoCustom(CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transform, None, transforms)
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, target = super().__getitem__(index)

        target_mask = self.coco.annToMask(target[0])
        for i in range(len(target)):
            target_mask += self.coco.annToMask(target[i])

        if self.target_transform is not None:
            target_mask = self.target_transform(target_mask)
        
        return image, target_mask


class Coco_DP(DatasetProvider):
    def __init__(self, reduced=False):
        super().__init__(reduced)
        self.transf = tf.Compose([
            tf.Resize(TARGET_SIZE),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.target_transf = tf.Compose([
            tf.ToPILImage(),
            tf.Resize(TARGET_SIZE),
            tf.ToTensor(),
        ])

    def create_train(self):
        return CocoCustom(root=COCO_TRAIN_PATH, annFile=COCO_TRAIN_ANNOTATIONS_PATH, \
                          transform=self.transf, target_transform=self.target_transf)

    def create_val(self):
        return CocoCustom(root=COCO_VAL_PATH, annFile=COCO_VAL_ANNOTATIONS_PATH, \
                          transform=self.transf, target_transform=self.target_transf)

    def create_test(self):
        return CocoCustom(root=COCO_TEST_PATH, \
                          transform=self.transf, target_transform=self.target_transf)

    def get_num_classes(self):
        return NUM_CLASSES
    
    def get_num_channels(self):
        return 3
