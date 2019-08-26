import torch

from options import DatasetOption, GlobalOption


def mask_nll_loss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask.byte()).mean()
    loss = loss.to(GlobalOption.device)
    return loss, nTotal.item()


def text_loss(batch_size, text_len, context, text_decoder, target_output,
              target_length, sos_id):
    context_hidden = context
    decoder_input = sos_id * torch.ones(batch_size, dtype=torch.long).view(1, -1).to(GlobalOption.device)

    # mask
    mask = torch.zeros(batch_size, text_len, dtype=torch.long).to(GlobalOption.device)
    mask.scatter_(1, (target_length - 1).view(-1, 1), 1)
    mask = 1 - mask.cumsum(dim=1)
    mask.scatter_(1, (target_length - 1).view(-1, 1), 1)
    mask.transpose_(0, 1)

    loss = 0
    n_totals = 0
    decoder_hidden = context_hidden.unsqueeze(0)
    for i in range(text_len):
        decoder_output, decoder_hidden = text_decoder(decoder_input, decoder_hidden)
        mask_loss, n_total = mask_nll_loss(decoder_output, target_output[i], mask)
        # output as the input of the next step
        decoder_input = target_output[i] * torch.ones(batch_size, dtype=torch.long).view(1, -1).to(GlobalOption.device)
        loss += mask_loss
        n_totals += n_total
    return loss / text_len, n_totals


def image_loss(batch_size, context, similarity,
               true_prod_text, true_prod_text_length, true_prod_image, true_prod_taxonomy, true_prod_attributes,
               false_prod_texts, false_prod_text_lengths, false_prod_images, false_images_num, false_prod_taxonomies,
               false_prod_attributess):
    ones = torch.ones(batch_size).to(GlobalOption.device)
    zeros = torch.zeros(batch_size).to(GlobalOption.device)
    true_cos_sim = similarity(context, true_prod_text, true_prod_text_length, true_prod_image, true_prod_taxonomy,
                              true_prod_attributes)
    # mask
    mask = torch.zeros(batch_size, DatasetOption.num_neg_images + 1, dtype=torch.long).to(GlobalOption.device)
    mask.scatter_(1, false_images_num.view(-1, 1), 1)
    mask = 1 - mask.cumsum(dim=1)
    mask = torch.narrow(mask, 1, 0, DatasetOption.num_neg_images)
    mask.transpose_(0, 1)

    losses = []
    for i in range(DatasetOption.num_neg_images):
        false_cos_sim = similarity(context, false_prod_texts[i], false_prod_text_lengths[i], false_prod_images[i],
                                   false_prod_taxonomies[i], false_prod_attributess[i])
        loss = torch.max(zeros, ones - true_cos_sim + false_cos_sim)
        losses.append(loss)
    losses = torch.stack(losses)
    # losses: (#img_per_utter, batch)
    loss = losses.masked_select(mask.byte()).mean()
    return loss
