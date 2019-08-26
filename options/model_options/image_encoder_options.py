from options.taxonomies import TaxonomyTree
from utils.better_abc import ABCMeta, abstract_attribute
from .mfb_fusion_options import ContextAttentionMFBFusionOption


class ImageEncoderOption(metaclass=ABCMeta):
    mfb_fusion = abstract_attribute()
    taxonomy_tree = abstract_attribute()
    in_channel_of_depth = abstract_attribute()
    out_channel_of_depth = abstract_attribute()
    kernel_size_of_depth = abstract_attribute()


class ContextImageEncoderOption(ImageEncoderOption):
    mfb_fusion = ContextAttentionMFBFusionOption()
    taxonomy_tree = TaxonomyTree()
    in_channel_of_depth = [64, 64, 32, 32, 64]
    out_channel_of_depth = [64, 32, 32, 64, 64]
    kernel_size_of_depth = [3, 3, 3, 3, 3]


class ProductImageEncoderOption(ImageEncoderOption):
    mfb_fusion = ContextAttentionMFBFusionOption()
    taxonomy_tree = TaxonomyTree()
    in_channel_of_depth = [64, 64, 32, 32, 64]
    out_channel_of_depth = [64, 32, 32, 64, 64]
    kernel_size_of_depth = [3, 3, 3, 3, 3]
