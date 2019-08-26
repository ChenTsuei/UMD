from .mfb_fusion_options import ContextMFBFusionOption


class ContextEncoderOption:
    input_size = ContextMFBFusionOption.outdim
    hidden_size = 1024
    num_layers = 1
    bidirectional = True

    hidden_linear_size = 1024
    each_input_linear_size = 1024

    use_linear_hidden = True
    use_linear_first_input = True
    use_linear_each_input = True
