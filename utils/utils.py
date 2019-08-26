import torch
from nltk import word_tokenize


def to_str(o):
    if isinstance(o, dict):
        res = []
        for key, val in o.items():
            res.append(key)
            res.append(to_str(val))
        return ' '.join(res)
    elif isinstance(o, list):
        return ' '.join([to_str(x) for x in o])
    elif isinstance(o, str):
        return o
    return str(o)


def pad_text(vocab, length, text):
    unk_id = vocab.get('<unk>')
    pad_id = vocab.get('<pad>')
    text = list(map(lambda w: vocab.get(w.lower(), unk_id), word_tokenize(text)))
    if len(text) > length - 1:
        text = text[:length - 1]
        text.append(vocab.get('</e>'))
        text_length = length
    else:
        text_length = len(text) + 1
        text.append(vocab.get('</e>'))
        text.extend([pad_id for _ in range(length - len(text))])
    return torch.tensor(text), text_length
