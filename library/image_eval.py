import torch

from options import DatasetOption
from options import GlobalOption


def image_eval(batch_size, context, similarity,
               true_prod_text, true_prod_text_length, true_prod_image, true_prod_taxonomy, true_prod_attributes,
               false_prod_texts, false_prod_text_lengths, false_prod_images, false_images_num, false_prod_taxonomies,
               false_prod_attributess):
    true_cos_sim = similarity(context, true_prod_text, true_prod_text_length, true_prod_image, true_prod_taxonomy,
                              true_prod_attributes)
    # mask
    mask = torch.zeros(batch_size, DatasetOption.num_neg_images + 1, dtype=torch.long).to(GlobalOption.device)
    mask.scatter_(1, false_images_num.view(-1, 1), 1)
    mask = 1 - mask.cumsum(dim=1)
    mask = torch.narrow(mask, 1, 0, DatasetOption.num_neg_images)
    mask.transpose_(0, 1)

    # number of negative images similarity greater than positive images
    gt = torch.zeros(batch_size, dtype=torch.long).to(GlobalOption.device)

    for i in range(DatasetOption.num_neg_images):
        false_cos_sim = similarity(context, false_prod_texts[i], false_prod_text_lengths[i], false_prod_images[i],
                                   false_prod_taxonomies[i], false_prod_attributess[i])
        gt += (false_cos_sim > true_cos_sim).long() * mask[i]
    return gt
