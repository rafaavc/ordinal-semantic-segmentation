import pytest
from datasets.ordinality_tree import OrdinalityTree


@pytest.mark.golden_test("golden/ordinality_tree/parse/*.yml")
def test_dict_parse_multi_ordinal(golden):
    tree = OrdinalityTree(golden['input'])
    print(tree)

    assert tree.is_multi_ordinal == golden.out['is_multi_ordinal']
    assert tree.contains_abstract_nodes == golden.out['contains_abstract_nodes']
    assert str(tree) == golden.out['output']


@pytest.mark.golden_test("golden/ordinality_tree/remove_abstract_nodes/*.yml")
def test_remove_abstract_nodes(golden):
    tree = OrdinalityTree(golden['input'])
    tree = tree.get_tree_without_abstract_nodes()
    print(tree)

    assert str(tree) == golden.out['output']
    assert tree.is_multi_ordinal == golden.out['is_multi_ordinal']
    assert not tree.contains_abstract_nodes

