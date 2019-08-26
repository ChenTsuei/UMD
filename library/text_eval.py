import torch
from options import GlobalOption


def greedy_decode(text_decoder, id2word, sos, eos, batch_size, text_len,
                  context, target_output):
    context_hidden = context
    decoder_input = sos * torch.ones(batch_size, dtype=torch.long).view(1,
                                                                        -1).to(
        GlobalOption.device)

    all_tokens = torch.zeros((text_len, batch_size), dtype=torch.long).to(
        GlobalOption.device)
    all_scores = torch.zeros((text_len, batch_size)).to(GlobalOption.device)

    decoder_hidden = context_hidden.unsqueeze(0)
    for i in range(text_len):
        decoder_output, decoder_hidden = text_decoder(decoder_input,
                                                      decoder_hidden)
        decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
        all_tokens[i], all_scores[i] = decoder_input, decoder_scores
        decoder_input = torch.unsqueeze(decoder_input, 0)
    for j in range(batch_size):
        str_pred = []
        str_true = []
        for i in range(text_len):
            if all_tokens[i][j] == eos:
                break
            word = id2word[all_tokens[i][j]]
            str_pred.append(word)

        for i in range(text_len):
            if target_output[i][j] == eos:
                break
            word = id2word[target_output[i][j]]
            str_true.append(word)
        print("{}\t{}".format(' '.join(str_pred), ' '.join(str_true)))
    return all_tokens
