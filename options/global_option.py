import torch


class GlobalOption:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
