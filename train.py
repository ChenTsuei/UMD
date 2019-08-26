import os
import sys
from collections import Counter
from datetime import datetime
from itertools import chain

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataset import RawData, MMDataset
from library import encode_context, preprocess_data, text_loss, image_loss
from models import TextEncoder, ImageEncoder, MFBFusion, ContextEncoder, \
    TextDecoder, Similarity
from options import GlobalOption, DatasetOption, TrainOption
from options.model_options import ContextMFBFusionOption, ContextEncoderOption
from options.model_options import ContextTextEncoderOption, \
    ContextImageEncoderOption
from options.model_options import TextDecoderOption, SimilarityOption
from test import test
from utils import get_embed_init
from valid import valid


def train(task, best_model_name):
    # daw dataset including train, valid, test
    raw_data = RawData(task, True, True, True)

    # construct dataset from raw data
    train_dataset = MMDataset(raw_data, task, 'train')
    valid_dataset = MMDataset(raw_data, task, 'valid')
    test_dataset = MMDataset(raw_data, task, 'test')

    # get vocab from raw_data
    vocab = raw_data.vocab
    vocab_size = len(vocab)
    sos_id = vocab.get('</s>')
    eos_id = vocab.get('</e>')

    # reverse index
    id2word = [None] * vocab_size
    for word, wid in vocab.items():
        id2word[wid] = word

    # train data loader
    data_loader = DataLoader(train_dataset, batch_size=TrainOption.batch_size,
                             shuffle=True,
                             num_workers=DatasetOption.num_data_loader_workers)

    # get GloVe from raw_data
    glove = raw_data.glove
    embed_init = get_embed_init(glove, vocab).to(GlobalOption.device)

    # construct common options
    context_text_encoder_option = ContextTextEncoderOption(vocab_size,
                                                           embed_init)
    context_image_encoder_option = ContextImageEncoderOption()
    context_fusion_option = ContextMFBFusionOption()
    context_encoder_option = ContextEncoderOption()

    # construct common models
    context_text_encoder = TextEncoder(context_text_encoder_option).to(
        GlobalOption.device)
    context_image_encoder = ImageEncoder(context_image_encoder_option).to(
        GlobalOption.device)
    context_mfb_fusion = MFBFusion(context_fusion_option).to(
        GlobalOption.device)
    context_encoder = ContextEncoder(context_encoder_option).to(
        GlobalOption.device)

    # common model parameters
    params = list(chain.from_iterable([list(model.parameters()) for model in [
        context_text_encoder,
        context_image_encoder,
        context_mfb_fusion,
        context_encoder
    ]]))

    # construct decoder for different tasks
    if task == 'text':
        # text decoder and its parameters
        text_decoder_option = TextDecoderOption(vocab_size, embed_init)
        text_decoder = TextDecoder(text_decoder_option).to(GlobalOption.device)
        params.extend(
            list(chain.from_iterable([list(model.parameters()) for model in [
                text_decoder
            ]])))
    elif task == 'image':
        # get product vocab from raw_data
        prod_vocab = raw_data.prod_vocab
        prod_vocab_size = len(prod_vocab)
        prod_sos_id = prod_vocab.get('</s>')

        # initialize embed
        prod_embed_init = get_embed_init(glove, prod_vocab).to(
            GlobalOption.device)

        # similarity model and its parameters
        similarity_option = SimilarityOption(prod_vocab_size, prod_sos_id,
                                             prod_embed_init)
        similarity = Similarity(similarity_option).to(GlobalOption.device)
        params.extend(
            list(chain.from_iterable([list(model.parameters()) for model in [
                similarity
            ]])))
    else:
        # task must be text or image
        raise ValueError('Task type error.')

    optimizer = Adam(params, lr=TrainOption.learning_rate)

    iteration = 0
    data_idx = 0
    sum_loss = 0
    min_valid_loss = None

    if task == 'image':
        max_recall = [None for _ in range(4)]

    # load best model
    best_model_file = os.path.join(DatasetOption.dump_root_directory,
                                   best_model_name)
    if os.path.isfile(best_model_file):
        state = torch.load(best_model_file)
        if task != state['task']:
            raise ValueError("task doesn't match.")

        iteration = state['iteration']
        min_valid_loss = state['valid_loss']

        # load common model parameters
        context_text_encoder.load_state_dict(state['context_text_encoder'])
        context_image_encoder.load_state_dict(state['context_image_encoder'])
        context_mfb_fusion.load_state_dict(state['context_mfb_fusion'])
        context_encoder.load_state_dict(state['context_encoder'])

        # load decoder parameters
        if task == 'text':
            text_decoder.load_state_dict(state['text_decoder'])
        elif task == 'image':
            similarity.load_state_dict(state['similarity'])

        # load optimizer parameters
        optimizer.load_state_dict(state['optimizer'])

    bad_loss_cnt = 0  # for patience
    for iter_id in range(iteration, TrainOption.num_iters):
        for data in data_loader:
            # switch to train mode
            context_text_encoder.train()
            context_image_encoder.train()
            context_mfb_fusion.train()
            context_encoder.train()
            if task == 'text':
                text_decoder.train()
            elif task == 'image':
                similarity.train()

            # initialize loss & grad
            loss = 0
            optimizer.zero_grad()

            # load and preprocess data
            batch_size, text_data, true_prods, false_prods = preprocess_data(
                task, data)
            texts, texts_lengths = text_data
            true_images, true_images_num, true_prod_text, true_prod_text_length, true_prod_taxonomy, true_prod_attributes = true_prods
            false_images, false_images_num, false_prod_text, false_prod_text_length, false_prod_taxonomy, false_prod_attributes = false_prods

            # encoded into a context vector
            context = encode_context(context_text_encoder,
                                     context_image_encoder, context_mfb_fusion,
                                     context_encoder,
                                     sos_id, batch_size, texts, texts_lengths,
                                     true_images,
                                     true_prod_taxonomy, true_prod_attributes)

            # decoders for different task
            if task == 'text':
                loss, _ = text_loss(batch_size, text_decoder_option.text_len,
                                    context,
                                    text_decoder,
                                    texts[-1, :, :].transpose(0, 1),
                                    texts_lengths[-1], sos_id)
                sum_loss += loss
            elif task == 'image':
                loss = image_loss(batch_size, context, similarity,
                                  true_prod_text[-1][0],
                                  true_prod_text_length[-1][0],
                                  true_images[-1][0], true_prod_taxonomy[-1][0],
                                  true_prod_attributes[-1][0],
                                  false_prod_text[-1],
                                  false_prod_text_length[-1],
                                  false_images[-1], false_images_num[-1],
                                  false_prod_taxonomy[-1],
                                  false_prod_attributes[-1])
                sum_loss += loss

            loss.backward()
            if task == 'text':
                clip_grad_norm_(params, max_norm=20)

            optimizer.step()
            data_idx += 1

            # print loss every TrainOption.print_freq batches
            if data_idx % TrainOption.print_freq == 0:
                cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sum_loss /= TrainOption.print_freq
                if task == 'text':
                    print(
                        'epoch: {} \tbatch: {} \tloss: {} \ttime: {}'.format(
                            iter_id, data_idx, sum_loss, cur_time))
                else:
                    print('epoch: {} \tbatch: {} \tloss: {} \ttime: {}'.format(
                        iter_id, data_idx, sum_loss, cur_time))
                # reset loss
                sum_loss = 0

            # valid every TrainOption.valid_freq batches
            if data_idx % TrainOption.valid_freq == 0:
                time1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if task == 'text':
                    valid_loss = valid(
                        task, context_text_encoder,
                        context_image_encoder,
                        context_mfb_fusion,
                        context_encoder, sos_id, valid_dataset,
                        text_decoder=text_decoder,
                        text_len=text_decoder_option.text_len)
                elif task == 'image':
                    valid_loss, recall = valid(task, context_text_encoder,
                                               context_image_encoder,
                                               context_mfb_fusion,
                                               context_encoder, sos_id,
                                               valid_dataset,
                                               similarity=similarity)

                    # save recall_{n}_{best_model_file} if recall-n is optimal
                    for i in range(4):
                        if max_recall[i] is None or recall[i] > max_recall[i]:
                            max_recall[i] = recall[i]
                            save_dict = {
                                'task': task,
                                'iteration': iter_id,
                                'valid_loss': valid_loss,
                                'recall': recall,
                                'context_text_encoder': context_text_encoder.state_dict(),
                                'context_image_encoder': context_image_encoder.state_dict(),
                                'context_mfb_fusion': context_mfb_fusion.state_dict(),
                                'context_encoder': context_encoder.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'similarity': similarity.state_dict()
                            }
                            torch.save(save_dict, os.path.join(
                                DatasetOption.dump_root_directory,
                                'recall_{}_'.format(i + 1) + best_model_name))
                    print(
                        'valid recall: {}/{}/{}/{}'.format(recall[0], recall[1],
                                                           recall[2],
                                                           recall[3]))

                time2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if task == 'text':
                    print(
                        'valid loss: {} \ttime: {} ~ {}'.format(
                            valid_loss,
                            time1, time2))
                else:
                    print('valid loss: {} \ttime: {} ~ {}'.format(valid_loss,
                                                                  time1, time2))

                # save if loss < min_valid_loss
                if min_valid_loss is None or valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    bad_loss_cnt = 0
                    save_dict = {
                        'task': task,
                        'iteration': iter_id,
                        'valid_loss': valid_loss,
                        'context_text_encoder': context_text_encoder.state_dict(),
                        'context_image_encoder': context_image_encoder.state_dict(),
                        'context_mfb_fusion': context_mfb_fusion.state_dict(),
                        'context_encoder': context_encoder.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    if task == 'text':
                        save_dict.update({
                            'text_decoder': text_decoder.state_dict()
                        })
                    elif task == 'image':
                        save_dict.update({
                            'recall': recall,
                            'similarity': similarity.state_dict()
                        })
                    torch.save(save_dict,
                               os.path.join(DatasetOption.dump_root_directory,
                                            best_model_name))
                else:
                    bad_loss_cnt += 1

                # test if out of patience
                if bad_loss_cnt >= TrainOption.patience:
                    if task == 'text':
                        loss, bleu = test(task, context_text_encoder,
                                          context_image_encoder,
                                          context_mfb_fusion,
                                          context_encoder, sos_id, test_dataset,
                                          text_decoder=text_decoder,
                                          text_len=text_decoder_option.text_len,
                                          id2word=id2word, eos_id=eos_id)
                        print('Text loss:', loss)
                        print(
                            'BLEU:{}/{}/{}/{}'.format(bleu[0], bleu[1], bleu[2],
                                                      bleu[3]))
                    elif task == 'image':
                        loss, recall = test(task, context_text_encoder,
                                            context_image_encoder,
                                            context_mfb_fusion,
                                            context_encoder, sos_id,
                                            test_dataset, similarity=similarity)
                        print('Image loss:', loss)
                        print('Recall:{}/{}/{}/{}'.format(recall[0], recall[1],
                                                          recall[2], recall[3]))


if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2])
