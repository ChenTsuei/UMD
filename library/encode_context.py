import torch

from options import DatasetOption, GlobalOption


def encode_context(context_text_encoder, context_image_encoder, context_mfb_fusion, context_encoder, sos_id, batch_size,
                   texts, texts_lengths, true_images, true_prod_taxonomy, true_prod_attributes):
    context = []
    start = sos_id * torch.ones(batch_size, dtype=torch.long).view(1, -1).to(GlobalOption.device)

    for i in range(DatasetOption.context_size):
        text, text_length = texts[i], texts_lengths[i]  # (batch, len), (batch, )
        text.transpose_(0, 1)  # (len, batch)
        text_with_sos = torch.cat((start, text), 0).to(GlobalOption.device)
        encoded_text = context_text_encoder(text_with_sos, text_length + 1)  # (batch, hidden_sz)
        for j in range(DatasetOption.num_pos_images):
            true_image = true_images[i][j]  # (batch, 3, img_sz, img_sz)
            prod_taxonomy = true_prod_taxonomy[i][j]  # (batch)
            prod_attributes = true_prod_attributes[i][j]  # (batch, attr)
            encoded_image = context_image_encoder(encoded_text, true_image, prod_taxonomy, prod_attributes)
            # (batch, img_enc_size)
            context.append(context_mfb_fusion(encoded_text, encoded_image))
    context = torch.stack(context).to(GlobalOption.device)  # (num_utters * imgs_per_utter, batch, mm_sz)
    context = context_encoder(context)  # (batch, mm_sz)
    return context
