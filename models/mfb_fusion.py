import torch
import torch.nn as nn

from options.model_options import MFBFusionOption
from .base_model import BaseModel
from options import GlobalOption


class MFBFusion(BaseModel):
    def __init__(self, mfb_fusion_option: MFBFusionOption):
        super(MFBFusion, self).__init__()
        self.out_dim = mfb_fusion_option.outdim
        self.linear1 = nn.Linear(mfb_fusion_option.text_size, mfb_fusion_option.joint_emb_size).to(GlobalOption.device)
        self.linear2 = nn.Linear(mfb_fusion_option.image_size, mfb_fusion_option.joint_emb_size).to(GlobalOption.device)
        self.dropout = nn.Dropout(p=mfb_fusion_option.dropout).to(GlobalOption.device)

    def forward(self, text_feat, image_feat):
        batch_size = text_feat.size(0)
        text_proj = self.linear1(text_feat)  # (batch, joint_emb_size)
        image_proj = self.linear2(image_feat)  # (batch, joint_emb_size)
        mm_eltwise = torch.mul(text_proj, image_proj)  # (batch, joint_emb_size)
        mm_drop = self.dropout(mm_eltwise)  # (batch, joint_emb_size)
        mm_resh = mm_drop.view(batch_size, 1, self.out_dim, -1)
        # (batch, 1, mfb_out_dim, mfb_factor_num)
        mm_sumpool = torch.sum(mm_resh, 3, keepdim=True)  # (batch, 1, mfb_out_dim, 1)
        mfb_out = torch.squeeze(mm_sumpool)  # (batch, mfb_out_dim, mfb_factor_num)
        return mfb_out
