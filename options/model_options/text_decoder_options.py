from .context_encoder_options import ContextEncoderOption


class TextDecoderOption:
    embed_size = 300
    context_size = ContextEncoderOption.hidden_linear_size
    hidden_size = 1024
    text_len = 30
    dropout = 0

    def __init__(self, vocab_size, embed_init=None):
        self.vocab_size = vocab_size
        self.embed_init = embed_init
