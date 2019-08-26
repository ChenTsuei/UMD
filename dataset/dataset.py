import pickle as pkl
import re
from os.path import join, isfile

import torch
from PIL import Image
from torch.utils import data

from dataset.data_models import Utterance, Item
from options import DatasetOption
from options.taxonomies import TaxonomyTree
from utils import pad_text


class MMDataset(data.Dataset):
    def __init__(self, raw_data, task, mode='train'):
        self.task = task
        self.mode = mode
        self.vocab = raw_data.vocab
        self.prod_vocab = raw_data.prod_vocab
        self.prod_collection = raw_data.prod_collection
        self.url2img = raw_data.url2img
        self.dialogs = getattr(raw_data, mode + '_dialogs')
        self.items = []
        self.text_pad_id = self.vocab.get('<pad>')
        self.text_eos_id = self.vocab.get('</e>')
        self.prod_text_pad_id = self.prod_vocab.get('<pad>')
        self.prod_text_eos_id = self.prod_vocab.get('</e>')
        self.empty_product = (
            torch.tensor([self.prod_text_eos_id] + [self.prod_text_pad_id] * (DatasetOption.product_text_length - 1)),
            1,
            TaxonomyTree.taxonomy_other_node.id,
            torch.tensor([node.id for node in TaxonomyTree.attribute_other_item_nodes], dtype=torch.long))
        # empty_product: (text, text_length, taxonomy, attributes)
        self.empty_utterance = (
            torch.tensor([self.text_eos_id] + [self.text_pad_id] * (DatasetOption.context_text_length - 1)),
            1,
            [''] * DatasetOption.num_pos_images,
            [None] * DatasetOption.num_pos_images,
            [''] * DatasetOption.num_neg_images,
            [None] * DatasetOption.num_neg_images)
        # empty_utterance: (text, text_length, images, products, false_images, false_products)
        self.empty_image = torch.zeros(3, DatasetOption.image_size, DatasetOption.image_size)
        self.get_items_from_dialogs()

    def __getitem__(self, index):
        item = self.items[index % len(self.items)]

        texts = []
        text_lengths = []
        true_prod_text = [[] for _ in range(DatasetOption.context_size + 1)]
        true_prod_text_length = [[] for _ in range(DatasetOption.context_size + 1)]
        true_prod_taxonomy = [[] for _ in range(DatasetOption.context_size + 1)]
        true_prod_attributes = [[] for _ in range(DatasetOption.context_size + 1)]
        true_images = [[] for _ in range(DatasetOption.context_size + 1)]
        true_images_num = [0] * (DatasetOption.context_size + 1)

        false_prod_text = [[] for _ in range(DatasetOption.context_size + 1)]
        false_prod_text_length = [[] for _ in range(DatasetOption.context_size + 1)]
        false_prod_taxonomy = [[] for _ in range(DatasetOption.context_size + 1)]
        false_prod_attributes = [[] for _ in range(DatasetOption.context_size + 1)]
        false_images = [[] for _ in range(DatasetOption.context_size + 1)]
        false_images_num = [0] * (DatasetOption.context_size + 1)

        # for each utterance in context
        for i in range(DatasetOption.context_size + 1):
            texts.append(item.texts[i])
            text_lengths.append(item.text_lengths[i])

            # true products
            for prod in item.true_prods[i]:
                if prod:
                    prod = prod.to_tensors(self.prod_vocab)
                else:
                    prod = self.empty_product
                true_prod_text[i].append(prod[0])
                true_prod_text_length[i].append(prod[1])
                true_prod_taxonomy[i].append(prod[2])
                true_prod_attributes[i].append(prod[3])

            true_prod_text[i] = torch.stack(true_prod_text[i])
            true_prod_text_length[i] = torch.tensor(true_prod_text_length[i])
            true_prod_taxonomy[i] = torch.tensor(true_prod_taxonomy[i])
            true_prod_attributes[i] = torch.stack(true_prod_attributes[i])

            for image in item.true_images[i]:
                image = self.url2img.get(image)
                if image is None:
                    continue
                path = join(DatasetOption.image_root_directory, image)
                if isfile(path):
                    try:
                        content = Image.open(join(DatasetOption.image_root_directory, image)).convert("RGB")
                    except OSError:
                        continue
                    true_images[i].append(DatasetOption.transform(content))
            true_images_num[i] = len(true_images[i])
            true_images[i] += [self.empty_image] * (DatasetOption.num_pos_images - true_images_num[i])
            true_images[i] = torch.stack(true_images[i])

            # false products
            for prod in item.false_prods[i]:
                if prod:
                    prod = prod.to_tensors(self.prod_vocab)
                else:
                    prod = self.empty_product
                false_prod_text[i].append(prod[0])
                false_prod_text_length[i].append(prod[1])
                false_prod_taxonomy[i].append(prod[2])
                false_prod_attributes[i].append(prod[3])

            false_prod_text[i] = torch.stack(false_prod_text[i])
            false_prod_text_length[i] = torch.tensor(false_prod_text_length[i])
            false_prod_taxonomy[i] = torch.tensor(false_prod_taxonomy[i])
            false_prod_attributes[i] = torch.stack(false_prod_attributes[i])

            for image in item.false_images[i]:
                image = self.url2img.get(image)
                if image is None:
                    continue
                path = join(DatasetOption.image_root_directory, image)
                if isfile(path):
                    try:
                        content = Image.open(join(DatasetOption.image_root_directory, image)).convert("RGB")
                    except OSError:
                        continue
                    false_images[i].append(DatasetOption.transform(content))
            false_images_num[i] = len(false_images[i])
            false_images[i] += [self.empty_image] * (DatasetOption.num_neg_images - false_images_num[i])
            false_images[i] = torch.stack(false_images[i])

        # convert to torch tensors
        texts = torch.stack(texts)
        text_lengths = torch.tensor(text_lengths)

        true_prod_text = torch.stack(true_prod_text)
        true_prod_text_length = torch.stack(true_prod_text_length)
        true_prod_taxonomy = torch.stack(true_prod_taxonomy)
        true_prod_attributes = torch.stack(true_prod_attributes)
        true_images = torch.stack(true_images)
        true_images_num = torch.tensor(true_images_num)

        false_prod_text = torch.stack(false_prod_text)
        false_prod_text_length = torch.stack(false_prod_text_length)
        false_prod_taxonomy = torch.stack(false_prod_taxonomy)
        false_prod_attributes = torch.stack(false_prod_attributes)
        false_images = torch.stack(false_images)
        false_images_num = torch.tensor(false_images_num)

        # pack into a tuple
        true_prods = true_images, true_images_num, true_prod_text, true_prod_text_length, true_prod_taxonomy, true_prod_attributes
        false_prods = false_images, false_images_num, false_prod_text, false_prod_text_length, false_prod_taxonomy, false_prod_attributes

        return texts, text_lengths, true_prods, false_prods

    def __len__(self):
        return len(self.items)

    def get_items_from_dialogs(self):
        # just load the item pkl if it exists
        item_pkl = getattr(DatasetOption, '{}_{}_item_pkl'.format(self.task, self.mode))
        if isfile(item_pkl):
            print('reading item pkl {}'.format(item_pkl))
            self.items = pkl.load(open(item_pkl, 'rb'))
            print('item pkl %s read complete' % item_pkl)
            return

        for item_idx, dialog in enumerate(self.dialogs):
            print('get items from dialogs {}/{}'.format(item_idx + 1, len(self.dialogs)))

            # standardize utterance
            # user, system, user, system...
            std_dialog = []
            for utter in dialog:
                if not std_dialog:
                    if utter.speaker != 'user':
                        std_dialog.append(Utterance('user', '', [], []))
                    else:
                        std_dialog.append(utter)
                else:
                    if utter.speaker != std_dialog[-1].speaker:
                        std_dialog.append(utter)
                    else:
                        std_dialog[-1].text += ' ' + utter.text
                        std_dialog[-1].images += utter.images
                        std_dialog[-1].false_images += utter.false_images

            item = [self.empty_utterance] * DatasetOption.context_size

            for idx, utter in enumerate(std_dialog):
                text, text_length = pad_text(self.vocab, DatasetOption.context_text_length, utter.text)
                true_images, true_prods = self.get_imgs_prods(utter.images, DatasetOption.num_pos_images)
                false_images, false_prods = self.get_imgs_prods(utter.false_images, DatasetOption.num_neg_images)
                item.append((text, text_length, true_images, true_prods, false_images, false_prods))
                if utter.speaker == 'system':
                    item = item[-(DatasetOption.context_size + 1):]
                    texts, text_lengths, true_images, true_prods, false_images, false_prods = map(list, zip(*item))
                    if self.task == 'image':
                        if self.has_no_image(true_images[-1]) or self.has_no_image(false_images[-1]):
                            continue
                    self.items.append(Item(texts, text_lengths, true_images, true_prods, false_images, false_prods))

        # save items to pkl file
        print('save item pkl to {}...'.format(item_pkl))
        with open(item_pkl, 'wb') as f:
            pkl.dump(self.items, f)
        print('saved')

    def is_valid_image(self, url):
        image = self.url2img.get(url)
        if image is None:
            return False
        path = join(DatasetOption.image_root_directory, image)
        if not isfile(path):
            return False
        try:
            Image.open(join(DatasetOption.image_root_directory, image)).convert("RGB")
        except OSError:
            return False
        return True

    # check if the image urls is valid
    def has_no_image(self, urls):
        for url in urls:
            if not self.is_valid_image(url):
                continue
            return False
        return True

    def get_imgs_prods(self, urls, img_num):
        if len(urls) > img_num:
            urls = urls[:img_num]
        elif len(urls) < img_num:
            urls += [''] * (img_num - len(urls))
        prods = [self.prod_collection.prod_from_url(url) for url in urls]
        return urls, prods
