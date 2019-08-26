import torch

from torch import nn
from torch.nn.functional import cosine_similarity
from options import GlobalOption
from .base_model import BaseModel
from .image_encoder import ImageEncoder
from .mfb_fusion import MFBFusion
from .text_encoder import TextEncoder


class Similarity(BaseModel):
    def __init__(self, similarity_option):
        super(Similarity, self).__init__()
        self.text_encoder = TextEncoder(similarity_option.text_encoder_option).to(GlobalOption.device)
        self.image_encoder = ImageEncoder(similarity_option.image_encoder_option).to(GlobalOption.device)
        self.mfb_fusion = MFBFusion(similarity_option.mfb_fusion_option).to(GlobalOption.device)
        self.sos_id = similarity_option.sos_id

    def forward(self, context, text, length, image, prod_taxonomy, prod_attributes):
        # context: (batch, context_hidden_size)

        # prepare data for text encoder
        batch_size = context.size(0)
        start = torch.ones(batch_size, dtype=torch.long).view(1, -1).to(GlobalOption.device)
        start = self.sos_id * start
        text.transpose_(0, 1)
        text_with_sos = torch.cat((start, text), 0).to(GlobalOption.device)

        encoded_text = self.text_encoder(text_with_sos, length + 1)
        encoded_image = self.image_encoder(encoded_text, image, prod_taxonomy, prod_attributes)
        encoded_mm = self.mfb_fusion(encoded_text, encoded_image)

        return cosine_similarity(context, encoded_mm)
