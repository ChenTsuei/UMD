from collections import defaultdict, deque

from options.taxonomies.taxonomy_tree_node import Node


def tree():
    return defaultdict(tree)


def add_taxonomy(t, path):
    for node in path:
        t = t[node]


def build_tree(raw_taxonomy_list, raw_attribute_list):
    raw_taxonomy_tree = tree()
    for taxonomy in raw_taxonomy_list:
        add_taxonomy(raw_taxonomy_tree, taxonomy)
    queue = deque()
    taxonomy_nodes = []
    last_layer_taxonomy_nodes = []
    attribute_nodes = []
    attribute_item_nodes = []
    taxonomy_other_node = None
    taxonomy_nodes.append(Node(0, 'root', 0, [], 0))
    queue.append((0, raw_taxonomy_tree))
    while len(queue) > 0:
        parent_id, parent_tree = queue.popleft()
        for name, current_tree in parent_tree.items():
            taxonomy_nodes.append(
                Node(len(taxonomy_nodes), name, parent_id, [],
                     taxonomy_nodes[parent_id].depth + 1))
            if name == '__other__':
                taxonomy_other_node = taxonomy_nodes[-1]
            taxonomy_nodes[parent_id].children.append(taxonomy_nodes[-1].id)
            queue.append((taxonomy_nodes[-1].id, current_tree))
    depth = taxonomy_nodes[-1].depth
    for node in taxonomy_nodes:
        if node.depth == depth:
            last_layer_taxonomy_nodes.append(node)
    for i, (name, items) in enumerate(raw_attribute_list):
        attribute_nodes.append(
            Node(i, name, None, [], last_layer_taxonomy_nodes[0].depth + 1))
        for item in items:
            item_idx = len(attribute_item_nodes)
            attribute_nodes[i].children.append(item_idx)
            attribute_item_nodes.append(
                Node(item_idx, item, i, None, attribute_nodes[i].depth + 1))
    attribute_other_item_nodes = [node for node in attribute_item_nodes if node.name == '__other__']
    return taxonomy_nodes, taxonomy_other_node, last_layer_taxonomy_nodes, attribute_nodes, attribute_item_nodes, attribute_other_item_nodes
