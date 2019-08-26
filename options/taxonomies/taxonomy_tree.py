from options.taxonomies.raw_attributes import raw_attributes
from options.taxonomies.raw_taxonomies import raw_taxonomies
from .helper import *


class TaxonomyTree:
    raw_taxonomy_list = [taxonomy_str.strip().split('>') for taxonomy_str in raw_taxonomies.split('\n')]
    raw_attribute_list = [(attribute[0], attribute[1].strip().split(';') + ['__other__']) for attribute in
                          [attribute_str.split(':') for attribute_str in raw_attributes.split('\n')]]
    taxonomy_nodes, taxonomy_other_node, last_layer_taxonomy_nodes, attribute_nodes, attribute_item_nodes, attribute_other_item_nodes = build_tree(
        raw_taxonomy_list, raw_attribute_list)

    @staticmethod
    def get_taxonomy_id(taxonomy_str):
        def helper(tax_str):
            res = None
            for node in TaxonomyTree.last_layer_taxonomy_nodes:
                for name in node.name.split(','):
                    cnt = tax_str.count(name)
                    if cnt > 0:
                        if res:
                            if cnt > res[1]:
                                res = (node.id, cnt)
                        else:
                            res = (node.id, cnt)
                            break
            return res

        taxonomy_str = taxonomy_str.lower()
        result = helper('/'.join([s for s in taxonomy_str.split('/') if s.find('&') == -1]))
        if not result:
            result = helper(taxonomy_str)
            if not result:
                result = helper('__other__')
        return result[0]

    @staticmethod
    def get_attribute_item_id(attribute, item_str):

        def helper(item_str):
            for node in TaxonomyTree.attribute_nodes:
                if node.name != attribute:
                    continue
                for child in node.children:
                    for s in TaxonomyTree.attribute_item_nodes[child].name.split(','):
                        cnt = item_str.count(s)
                        if cnt > 0:
                            return child
                    if TaxonomyTree.attribute_item_nodes[child].name == '__other__':
                        return child

        item_str = item_str.lower()
        res = helper(item_str)
        return res
