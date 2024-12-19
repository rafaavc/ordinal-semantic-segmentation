import pytest, torch
from datasets.ordinality_tree import OrdinalityTree
from losses.multi_co2 import get_unimodal_operations, get_unimodal_operations_str
from metrics import UnimodalPercentage


@pytest.mark.golden_test("golden/unimodality/ops/*.yml")
def test_unimodality_ops(golden):
    ops = get_unimodal_operations(OrdinalityTree(golden['input']))

    assert get_unimodal_operations_str(ops) == golden.out['output']


@pytest.mark.golden_test("golden/metric/*.yml")
def test_unimodality_metric(golden):
    input = torch.tensor(golden['input'])
    tree = OrdinalityTree(golden['tree'])

    unimodal = UnimodalPercentage(tree)
    unimodal.update(input, None)

    assert float(unimodal.compute()) == golden.out['unimodality_metric']
