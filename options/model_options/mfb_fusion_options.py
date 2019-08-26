from utils.better_abc import ABCMeta, abstract_attribute
from .text_encoder_options import ContextTextEncoderOption


class MFBFusionOption(metaclass=ABCMeta):
    text_size = abstract_attribute()
    image_size = abstract_attribute()
    joint_emb_size = abstract_attribute()
    dropout = abstract_attribute()
    outdim = abstract_attribute()


class ContextMFBFusionOption(MFBFusionOption):
    text_size = ContextTextEncoderOption.hidden_size * (2 if ContextTextEncoderOption.bidirectional else 1)
    image_size = 4096
    joint_emb_size = 1024
    dropout = 0
    outdim = 1024

    def __init__(self):
        super(ContextMFBFusionOption, self).__init__()
        pass


class ContextAttentionMFBFusionOption(MFBFusionOption):
    text_size = ContextTextEncoderOption.hidden_size * (2 if ContextTextEncoderOption.bidirectional else 1)
    image_size = 4096
    joint_emb_size = 1024
    dropout = 0
    outdim = 1

    def __init__(self):
        super(ContextAttentionMFBFusionOption, self).__init__()


class ProductMFBFusionOption(MFBFusionOption):
    text_size = ContextTextEncoderOption.hidden_size * (2 if ContextTextEncoderOption.bidirectional else 1)
    image_size = 4096
    joint_emb_size = 1024
    dropout = 0
    outdim = 1024

    def __init__(self):
        super(ProductMFBFusionOption, self).__init__()


class ProductAttentionMFBFusionOption(MFBFusionOption):
    text_size = ContextTextEncoderOption.hidden_size * (2 if ContextTextEncoderOption.bidirectional else 1)
    image_size = 4096
    joint_emb_size = 1024
    dropout = 0
    outdim = 1

    def __init__(self):
        super(ProductAttentionMFBFusionOption, self).__init__()
