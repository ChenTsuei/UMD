import torch
from torch.utils.data import DataLoader

from library import encode_context, preprocess_data, text_loss, image_loss, \
    image_eval, greedy_decode
from options import DatasetOption, TestOption


def test(task, context_text_encoder, context_image_encoder, context_mfb_fusion,
         context_encoder, sos_id, dataset,
         text_decoder=None, text_len=None, similarity=None, id2word=None,
         eos_id=None):
    data_loader = DataLoader(dataset, batch_size=TestOption.batch_size,
                             shuffle=False,
                             num_workers=DatasetOption.num_data_loader_workers)
    context_text_encoder.eval()
    # context_image_encoder.eval()
    context_mfb_fusion.eval()
    context_encoder.eval()

    if task == 'text':
        text_decoder.eval()
        num_texts = 0
    elif task == 'image':
        similarity.eval()
        num_images = 0
        num_rank = [0 for _ in range(5)]

    loss = 0
    num_iter = 0
    with torch.no_grad():
        for data_idx, data in enumerate(data_loader):
            num_iter += 1
            batch_size, text_data, true_prods, false_prods = preprocess_data(
                task, data)
            texts, texts_lengths = text_data
            true_images, true_images_num, true_prod_text, true_prod_text_length, true_prod_taxonomy, true_prod_attributes = true_prods
            false_images, false_images_num, false_prod_text, false_prod_text_length, false_prod_taxonomy, false_prod_attributes = false_prods
            context = encode_context(context_text_encoder,
                                     context_image_encoder, context_mfb_fusion,
                                     context_encoder,
                                     sos_id, batch_size, texts, texts_lengths,
                                     true_images,
                                     true_prod_taxonomy, true_prod_attributes)
            if task == 'text':
                num_texts += batch_size
                l, _ = text_loss(batch_size, text_len, context,
                                 text_decoder,
                                 texts[-1, :, :].transpose(0, 1),
                                 texts_lengths[-1], sos_id)
                greedy_decode(text_decoder, id2word, sos_id, eos_id, batch_size,
                              text_len, context, texts[-1, :, :].transpose(0, 1))
                loss += l
            elif task == 'image':
                num_images += batch_size
                l = image_loss(batch_size, context, similarity,
                               true_prod_text[-1][0],
                               true_prod_text_length[-1][0],
                               true_images[-1][0], true_prod_taxonomy[-1][0],
                               true_prod_attributes[-1][0],
                               false_prod_text[-1], false_prod_text_length[-1],
                               false_images[-1], false_images_num[-1],
                               false_prod_taxonomy[-1],
                               false_prod_attributes[-1])
                gt = image_eval(batch_size, context, similarity,
                                true_prod_text[-1][0],
                                true_prod_text_length[-1][0],
                                true_images[-1][0], true_prod_taxonomy[-1][0],
                                true_prod_attributes[-1][0],
                                false_prod_text[-1], false_prod_text_length[-1],
                                false_images[-1], false_images_num[-1],
                                false_prod_taxonomy[-1],
                                false_prod_attributes[-1])
                for i in range(batch_size):
                    num_rank[gt[i]] += 1
                loss += l

    context_text_encoder.train()
    context_image_encoder.train()
    context_mfb_fusion.train()
    context_encoder.train()
    if task == 'text':
        text_decoder.train()
        return loss / num_iter
    elif task == 'image':
        similarity.train()

        temp = 0
        recall = []
        for i in range(4):
            temp += num_rank[i]
            recall.append(temp / num_images)
        return loss / num_iter, tuple(recall)
