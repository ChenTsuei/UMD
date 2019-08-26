import torch
import torch.nn as nn
import torch.nn.functional as F

from options.model_options.text_encoder_options import TextEncoderOption
from .base_model import BaseModel
from options import GlobalOption


class TextEncoder(BaseModel):
    def __init__(self, text_encoder_option: TextEncoderOption):
        super(TextEncoder, self).__init__()
        self.hidden_size = text_encoder_option.hidden_size
        self.num_dir = 2 if text_encoder_option.bidirectional else 1
        self.embedding = nn.Embedding(text_encoder_option.vocab_size, text_encoder_option.embed_size,
                                      padding_idx=text_encoder_option.padding_idx).to(GlobalOption.device)
        if text_encoder_option.embed_init is not None:
            self.embedding = self.embedding.from_pretrained(text_encoder_option.embed_init)
        self.rnn = nn.LSTM(input_size=text_encoder_option.embed_size, hidden_size=text_encoder_option.hidden_size,
                           num_layers=text_encoder_option.num_layers, bidirectional=text_encoder_option.bidirectional,
                           batch_first=False, dropout=text_encoder_option.dropout).to(GlobalOption.device)
        self.conv1 = nn.Conv2d(text_encoder_option.hidden_size * self.num_dir, text_encoder_option.conv_size, 1).to(GlobalOption.device)
        self.conv2 = nn.Conv2d(text_encoder_option.conv_size, 1, 1).to(GlobalOption.device)
        self.softmax = nn.Softmax(dim=1).to(GlobalOption.device)

    def forward(self, input_seq, input_lengths):
        length = input_seq.size(0)
        embedded = self.embedding(input_seq)  # (seq_len, batch)
        rnned, _ = self.rnn(embedded)
        # (seq_len, batch, num_directions * hidden_size)
        rnned = rnned.permute(1, 2, 0)  # (batch, num_directions * hidden_size, seq_len)
        conved = self.conv1(torch.unsqueeze(rnned, 3))  # (batch, conv_size, seq_len, 1)
        conved = F.relu(conved)  # (batch, conv_size, seq_len, 1)
        conved = self.conv2(conved)  # (batch, 1, seq_len, 1)
        conved = conved.view(-1, length)  # (batch, seq_len)
        x_softmax = self.softmax(conved)
        x_softmax = x_softmax.view(-1, 1, length)  # (batch, 1, seq_len)
        output = torch.mul(x_softmax, rnned)  # (batch, num_directions * hidden_size, seq_len)
        output = torch.sum(output, 2)  # (batch, num_directions * hidden_size)
        return output
