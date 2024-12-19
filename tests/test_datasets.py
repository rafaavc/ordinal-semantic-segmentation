import pytest
from utils import custom_import as imp
from datasets import DatasetProvider


@pytest.mark.golden_test("golden/datasets/*.yml")
def test_contact_surface_ops(golden):
    dataset_provider = imp.custom_import_class(golden['dataset'], 'dataset')

    scale1: DatasetProvider = dataset_provider(1, "wroadagents1_nodrivable")
    scale_5: DatasetProvider = dataset_provider(.5, "wroadagents1_nodrivable")

    assert len(scale1.create_train()) == golden.out['scale1_train_len']
    assert len(scale_5.create_train()) == golden.out['scale_5_train_len']

    if "Unsupervised" not in golden['dataset']:
        assert len(scale1.create_test()) == golden.out['scale1_test_len']
        assert len(scale_5.create_test()) == golden.out['scale_5_test_len']

        assert len(scale1.create_val()) == golden.out['scale1_val_len']
        assert len(scale_5.create_val()) == golden.out['scale_5_val_len']

        mask_val = scale1.create_val()[0][1]
        assert (None if mask_val is None else str(mask_val.tolist())) == golden.out['scale1_val_mask0']

        mask_test = scale1.create_test()[0][1]
        assert (None if mask_test is None else str(mask_test.tolist())) == golden.out['scale1_test_mask0']
    
