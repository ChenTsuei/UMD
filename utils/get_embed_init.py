import torch


def get_embed_init(glove, vocab):
    embed = [None] * len(vocab)
    for word, idx in vocab.items():
        vec = glove.get(word)
        if vec is None:
            vec = torch.zeros(300)
            vec.uniform_(-0.25, 0.25)
        else:
            vec = torch.tensor(vec)
        embed[idx] = vec
    embed = torch.stack(embed)
    return embed
