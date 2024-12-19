from typing import Union
import copy


class OrdinalityTreeNode():
    def __init__(self, value, parent=None, is_abstract=False):
        self.value = value
        self.parent: Union[OrdinalityTreeNode, None] = parent
        self.children: list[OrdinalityTreeNode] = []
        self.is_abstract: bool = is_abstract
    
    def __str__(self) -> str:
        abstract = ", abstract" if self.is_abstract else ""
        return f"[{self.value}{abstract}]"


OrdinalityTreeDict = dict[Union[int, str], Union[dict, None]]


def count_int_keys(dict):
    result = 0
    for k in dict.keys():
        if type(k) == int:
            result += 1
    return result


class TreeLevelNodeTracker():
    def __init__(self):
        self.dict = {}

    def add(self, node: OrdinalityTreeNode, level: int):
        if level not in self.dict:
            self.dict[level] = []

        self.dict[level].append(node)

    def merge(self, tracker):
        for level, nodes_list in tracker.get_dict_copy().items():
            for n in nodes_list:
                self.add(n, level)

    def pop(self, level: int):
        if level not in self.dict:
            return
        
        self.dict[level].pop()
        if len(self.dict[level]) == 0:
            del self.dict[level]

    def get_dict_copy(self):
        return { level: [*nodes] for level, nodes in self.dict.items() }


class OrdinalityTree():
    """
    Accepts an ordinality tree dictionary. Example:
        {
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
        }
    """
    def __init__(self, tree: OrdinalityTreeDict):
        self.is_multi_ordinal = False
        self.contains_abstract_nodes = False
        self.dict_tree = copy.deepcopy(tree)
        self.parse_tree(tree)


    def parse_tree(self, tree: OrdinalityTreeDict):
        assert len(tree) == 1, "Ordinality tree dictionary contains more than two roots" # currently not supporting more than one root

        value = next(iter(tree.keys()))
        branch = tree[value]
        self.root = OrdinalityTreeNode(value)

        self.parse_branch(branch, self.root)


    def parse_branch(self, branch: OrdinalityTreeDict, parent: OrdinalityTreeNode):
        if count_int_keys(branch) > 1:
            self.is_multi_ordinal = True

        for child_value in branch.keys():
            if type(child_value) != int:
                continue

            sub_branch = branch[child_value]

            is_abstract = sub_branch is not None and 'is_abstract' in sub_branch.keys()
            self.contains_abstract_nodes |= is_abstract

            child = OrdinalityTreeNode(child_value, parent=parent, is_abstract=is_abstract)
            parent.children.append(child)

            if sub_branch is None or count_int_keys(sub_branch) == 0:
                continue

            self.parse_branch(sub_branch, child)

    
    def get_tree_without_abstract_nodes(self):
        if not self.contains_abstract_nodes:
            return self
        
        dict_tree = {}
        self.convert_to_non_abstract_dict_tree(dict_tree, self.root)

        return OrdinalityTree(dict_tree)


    def convert_to_non_abstract_dict_tree(self, dict_parent, current_node: OrdinalityTreeNode):
        if current_node.children:
            if not current_node.is_abstract:
                dict_parent[current_node.value] = {}
                child_parent = dict_parent[current_node.value]
            else:
                child_parent = dict_parent

            for child in current_node.children:
                self.convert_to_non_abstract_dict_tree(child_parent, child)
            return

        if not current_node.is_abstract:
            dict_parent[current_node.value] = None        


    def visit(self, do_custom_action, ascendants: Union[TreeLevelNodeTracker, None]=None,   
              current_node: Union[OrdinalityTreeNode, None]=None, current_level: int=0):
        if current_node is None:
            current_node = self.root
        if ascendants is None:
            ascendants = TreeLevelNodeTracker()

        children_output = []
        ascendants.add(current_node, current_level)
        for child in current_node.children:
            children_output.append(self.visit(do_custom_action, ascendants=ascendants, current_node=child, \
                       current_level=current_level+1))
        ascendants.pop(current_level)

        return do_custom_action(current_node=current_node, ascendants=ascendants, children_output=children_output, current_level=current_level)


    def get_str(self, current_node, indent=0):
        result = ""
        if current_node is None:
            current_node = self.root

        result += indent * " " + str(current_node) + "\n"
        
        for child in current_node.children:
            result += self.get_str(child, indent + 4)
        
        return result


    def __str__(self) -> str:
        return self.get_str(self.root)
