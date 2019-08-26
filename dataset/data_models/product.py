import torch

from options import DatasetOption
from options.taxonomies import TaxonomyTree
from utils import pad_text, to_str


class Product:
    def __init__(self, attr_dict=None):
        if attr_dict is None:
            attr_dict = dict()

        # convert structured product data into a string
        self.prod_str = to_str(
            {key: val for key, val in attr_dict.items() if
             key not in DatasetOption.product_exclude_attributes and key not in DatasetOption.product_attributes})

        # set product attributes such as url...
        for key in DatasetOption.product_attributes:
            setattr(self, key, attr_dict.get(key, None))

        # get taxonomy
        self.taxonomy = TaxonomyTree.get_taxonomy_id(attr_dict.get('taxonomy', ''))

        # get value for each attribute
        self.attributes = [
            TaxonomyTree.get_attribute_item_id(attribute_node.name, attr_dict.get(attribute_node.name, ''))
            for attribute_node in TaxonomyTree.attribute_nodes]

    def to_tensors(self, vocab):
        # convert a product into a tensor
        text, length = pad_text(vocab, DatasetOption.product_text_length, self.prod_str)
        attributes = torch.tensor(self.attributes, dtype=torch.long)
        return text, length, self.taxonomy, attributes
