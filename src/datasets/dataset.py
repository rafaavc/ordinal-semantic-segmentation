import random
from torch.utils.data import Dataset as PyTorchDataset
from datasets.ordinality_tree import OrdinalityTree

class DatasetProvider():
    def __init__(self, scale: float, K: int):
        assert scale > 0 and scale <= 1
        self.scale = scale
        self.K = K

    def create_train(self) -> PyTorchDataset:
        pass

    def create_val(self) -> PyTorchDataset:
        raise ValueError("Trying to create validation dataset from dataset that doesn't support it.")

    def create_test(self) -> PyTorchDataset:
        pass

    def create_unsupervised(self) -> PyTorchDataset:
        raise ValueError("Trying to create unsupervised dataset from dataset that doesn't support it.")

    def get_num_classes(self) -> int:
        return self.K

    def get_num_channels(self) -> int:
        pass

    def get_ordinality_tree(self) -> OrdinalityTree:
        raise ValueError("Trying to get ordinality tree from dataset that doesn't support it.")


class Dataset(PyTorchDataset):
    def __init__(self) -> None:
        super().__init__()
        self.random_seeded = random.Random(1)

    def get_train_test_split(self, files: list[str]):
        files = [*files]
        dataset_len = len(files)
        train_len = int(dataset_len * 0.8)

        self.random_seeded.shuffle(files)
        train, test = files[:train_len], files[train_len:]

        print(f"train: {len(train)}, test: {len(test)}")

        assert len(set(train) & set(test)) == 0
        assert len(train) + len(test) == dataset_len

        return train, test

    def set_files(self, files: list[str], split: str):
        files = sorted(files) # sort for guarantees that it is repeatable
        train, test = self.get_train_test_split(files)
        self.files = train if split == "train" else test


class SampledDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
    def sample_files(self, dataset_scale: float, split: str):
        assert split in ('train', 'test')
        if split == "test": # don't sample the test dataset
            return
        
        # Dataset scale sampling
        original_size = len(self.files) 
        self.files = self.random_seeded.sample(self.files, int(dataset_scale * original_size))

        print(f"Sampled dataset size = {dataset_scale} * {original_size} = {len(self.files)}")
    
