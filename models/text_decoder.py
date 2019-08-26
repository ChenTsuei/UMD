import torch
import torch.nn as nn

from options import GlobalOption
from options.model_options import TextDecoderOption


class TextDecoder(nn.Module):
    def __init__(self, text_decoder_option: TextDecoderOption):
        super(TextDecoder, self).__init__()
        self.hidden_size = text_decoder_option.hidden_size
        self.embedding = nn.Embedding(text_decoder_option.vocab_size, text_decoder_option.embed_size).to(
            GlobalOption.device)
        if text_decoder_option.embed_init is not None:
            self.embedding.from_pretrained(text_decoder_option.embed_init)
        self.gru = nn.GRU(text_decoder_option.embed_size, text_decoder_option.hidden_size).to(GlobalOption.device)
        self.out = nn.Linear(text_decoder_option.hidden_size, text_decoder_option.vocab_size).to(GlobalOption.device)
        self.softmax = nn.Softmax(dim=1).to(GlobalOption.device)

    def forward(self, word, hidden):
        batch_size = hidden.size(1)
        embed = self.embedding(word[0]).view(1, batch_size, -1)
        output, hidden = self.gru(embed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
