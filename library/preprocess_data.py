from options import GlobalOption


def preprocess_data(task, data):
    texts, texts_lengths, true_prods, false_prods = data

    true_images, true_images_num, true_prod_text, true_prod_text_length, true_prod_taxonomy, true_prod_attributes = true_prods
    false_images, false_images_num, false_prod_text, false_prod_text_length, false_prod_taxonomy, false_prod_attributes = false_prods

    batch_size = texts.size(0)

    # to device
    texts = texts.to(GlobalOption.device)
    texts_lengths = texts_lengths.to(GlobalOption.device)
    true_images = true_images.to(GlobalOption.device)
    true_images_num = true_images_num.to(GlobalOption.device)
    true_prod_taxonomy = true_prod_taxonomy.to(GlobalOption.device)
    true_prod_attributes = true_prod_attributes.to(GlobalOption.device)

    if task == 'image':
        true_prod_text = true_prod_text.to(GlobalOption.device)
        true_prod_text_length = true_prod_text_length.to(GlobalOption.device)
        false_images = false_images.to(GlobalOption.device)
        false_images_num = false_images_num.to(GlobalOption.device)
        false_prod_text = false_prod_text.to(GlobalOption.device)
        false_prod_text_length = false_prod_text_length.to(GlobalOption.device)
        false_prod_taxonomy = false_prod_taxonomy.to(GlobalOption.device)
        false_prod_attributes = false_prod_attributes.to(GlobalOption.device)

    texts.transpose_(0, 1)  # (utter, batch, len)
    texts_lengths.transpose_(0, 1)  # (utter, batch)
    true_images.transpose_(0, 1)
    true_images.transpose_(1, 2)
    # (utter, imgs_per_utter, batch, 3, img_sz, img_sz)
    true_images_num.transpose_(0, 1)
    true_prod_taxonomy.transpose_(0, 1)
    true_prod_taxonomy.transpose_(1, 2)
    # (utter, imgs_per_utter, batch)
    true_prod_attributes.transpose_(0, 1)
    true_prod_attributes.transpose_(1, 2)
    # (utter, imgs_per_utter, batch, attr)

    if task == 'image':
        true_prod_text.transpose_(0, 1)
        true_prod_text.transpose_(1, 2)
        # (utter, imgs_per_utter, batch, len)
        true_prod_text_length.transpose_(0, 1)
        true_prod_text_length.transpose_(1, 2)
        # (utter, imgs_per_utter, batch)

        false_prod_text.transpose_(0, 1)
        false_prod_text.transpose_(1, 2)
        # (utter, num_neg_images, batch, len)
        false_prod_text_length.transpose_(0, 1)
        false_prod_text_length.transpose_(1, 2)
        # (utter, num_neg_images, batch)

        false_images.transpose_(0, 1)
        false_images.transpose_(1, 2)
        # (utter, num_neg_images, batch, 3, img_sz, img_sz)
        false_images_num.transpose_(0, 1)
        false_prod_taxonomy.transpose_(0, 1)
        false_prod_taxonomy.transpose_(1, 2)
        # (utter, imgs_per_utter, batch)
        false_prod_attributes.transpose_(0, 1)
        false_prod_attributes.transpose_(1, 2)
        # (utter, num_neg_images, batch, attr)

    return batch_size, (texts, texts_lengths), \
           (true_images, true_images_num, true_prod_text, true_prod_text_length, true_prod_taxonomy,
            true_prod_attributes), \
           (false_images, false_images_num, false_prod_text, false_prod_text_length, false_prod_taxonomy,
            false_prod_attributes)
