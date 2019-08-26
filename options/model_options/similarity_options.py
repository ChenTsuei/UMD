from .text_encoder_options import ProductTextEncoderOption
from .image_encoder_options import ProductImageEncoderOption
from .mfb_fusion_options import ProductMFBFusionOption
from .context_encoder_options import ContextEncoderOption


class SimilarityOption:
    image_encoder_option = ProductImageEncoderOption()
    mfb_fusion_option = ProductMFBFusionOption()
    context_hidden_size = ContextEncoderOption.hidden_size * (2 if ContextEncoderOption.bidirectional else 1)

    def __init__(self, vocab_size, sos_id, embed_init=None):
        self.vocab_size = vocab_size
        self.sos_id = sos_id
        self.embed_init = embed_init
        self.text_encoder_option = ProductTextEncoderOption(vocab_size, embed_init)
