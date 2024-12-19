from typing import Any
from torch.utils.data import Dataset, DataLoader, RandomSampler, SubsetRandomSampler


def get_test_dataloader(TEST_ARGS: Any, dataset: Dataset):
    kwargs = {'num_workers': 4, 'pin_memory': True} if TEST_ARGS.use_cuda else {'num_workers': 0}
    return DataLoader(dataset, batch_size=TEST_ARGS.batch_size, shuffle=False, **kwargs)


def get_train_val_dataloaders(TRAIN_ARGS: Any, train: Dataset, val: Dataset, train_idx, val_idx):
    kwargs = {'num_workers': 4, 'pin_memory': True} if TRAIN_ARGS.use_cuda else {'num_workers': 0}
    return DataLoader(train, batch_size=TRAIN_ARGS.batch_size, sampler=SubsetRandomSampler(train_idx), **kwargs), \
        DataLoader(val, batch_size=TRAIN_ARGS.batch_size, sampler=SubsetRandomSampler(val_idx), **kwargs)


def get_train_dataloader(TRAIN_ARGS: Any, train: Dataset):
    kwargs = {'num_workers': 4, 'pin_memory': True} if TRAIN_ARGS.use_cuda else {'num_workers': 0}
    return DataLoader(train, batch_size=TRAIN_ARGS.batch_size, sampler=RandomSampler(train), **kwargs)
