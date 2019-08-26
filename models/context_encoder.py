import torch.nn as nn

from options import GlobalOption
from options.model_options import ContextEncoderOption
from .base_model import BaseModel


class ContextEncoder(BaseModel):
    def __init__(self, context_encoder_option: ContextEncoderOption):
        super(ContextEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=ContextEncoderOption.input_size, hidden_size=ContextEncoderOption.hidden_size,
                           num_layers=ContextEncoderOption.num_layers, bidirectional=ContextEncoderOption.bidirectional,
                           batch_first=False).to(GlobalOption.device)

        self.linear_hidden = nn.Linear(
            ContextEncoderOption.hidden_size * (2 if ContextEncoderOption.bidirectional else 1),
            ContextEncoderOption.hidden_linear_size).to(GlobalOption.device)

    def forward(self, mms):
        _, (h_n, _) = self.rnn(mms)
        # (num_directions, batch, hidden_size)
        batch_size = h_n.size(1)
        h_n = h_n.transpose(0, 1)
        h_n = h_n.contiguous().view(batch_size, -1)
        hidden = self.linear_hidden(h_n)
        return hidden
