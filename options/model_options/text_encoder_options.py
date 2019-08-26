from utils.better_abc import ABCMeta, abstract_attribute


class TextEncoderOption(metaclass=ABCMeta):
    vocab_size = abstract_attribute()
    embed_init = abstract_attribute()
    padding_idx = abstract_attribute()
    embed_size = abstract_attribute()
    hidden_size = abstract_attribute()
    bidirectional = abstract_attribute()
    num_layers = abstract_attribute()
    dropout = abstract_attribute()
    conv_size = abstract_attribute()


class ContextTextEncoderOption(TextEncoderOption):
    padding_idx = 3
    embed_size = 300
    hidden_size = 1024
    bidirectional = True
    num_layers = 1
    dropout = 0
    conv_size = 128

    def __init__(self, vocab_size, embed_init=None):
        super(ContextTextEncoderOption, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init


class ProductTextEncoderOption(TextEncoderOption):
    padding_idx = 3
    embed_size = 300
    hidden_size = 1024
    bidirectional = True
    num_layers = 1
    dropout = 0
    conv_size = 128

    def __init__(self, vocab_size, embed_init=None):
        super(ProductTextEncoderOption, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init
