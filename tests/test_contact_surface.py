import pytest, torch
from datasets.ordinality_tree import OrdinalityTree
from losses.multi_cs_tv_v4 import get_contact_surface_operations, get_contact_surface_operations_str
from metrics import OrdinalContactSurface


@pytest.mark.golden_test("golden/metric/*.yml")
def test_contact_surface_metric(golden):
    input = torch.tensor(golden['input'])
    tree = OrdinalityTree(golden['tree'])

    contact_surface = OrdinalContactSurface(tree)
    contact_surface.update(input, None)

    assert float(contact_surface.compute()) == golden.out['contact_surface_metric']


@pytest.mark.golden_test("golden/contact_surface/ops/*.yml")
def test_contact_surface_ops(golden):
    ops = get_contact_surface_operations(OrdinalityTree(golden['input']))

    assert get_contact_surface_operations_str(ops) == golden.out['output']


@pytest.mark.golden_test("golden/contact_surface/ops/*.yml")
def test_contact_surface_ops(golden):
    ops = get_contact_surface_operations(OrdinalityTree(golden['input']), level_threshold=0)

    assert get_contact_surface_operations_str(ops) == golden.out['output_0']

