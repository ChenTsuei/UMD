import json
import os
import pickle as pkl
from collections import Counter
from os.path import isdir, isfile

from nltk import word_tokenize

from dataset.data_models import Utterance, Product, ProductCollection
from options import DatasetOption


class RawData:
    def __init__(self, task, train=True, valid=True, test=False):
        self.task = task
        # check if pkl or raw dataset exists
        if train and not isdir(DatasetOption.train_dialog_data_directory) and not isfile(DatasetOption.train_pkl):
            raise ValueError("No training dataset.")
        if valid and not isdir(DatasetOption.valid_dialog_data_directory) and not isfile(DatasetOption.valid_pkl):
            raise ValueError("No validation dataset.")
        if test and not isdir(DatasetOption.test_dialog_data_directory) and not isfile(DatasetOption.test_pkl):
            raise ValueError("No testing dataset.")

        self.vocab = None
        if train:
            self.train_dialogs = []
        if valid:
            self.valid_dialogs = []
        if test:
            self.test_dialogs = []

        # get vocab
        if not isfile(DatasetOption.vocab_pkl):
            if isfile(DatasetOption.train_pkl) or isfile(DatasetOption.valid_pkl) or isfile(DatasetOption.test_pkl):
                raise ValueError("Vocab doesn't exist but pkl dataset exists.")
            self.get_vocab()
        else:
            self.vocab = pkl.load(open(DatasetOption.vocab_pkl, 'rb'))

        # get train dialogs
        if train:
            if not isfile(DatasetOption.train_pkl):
                if not self.train_dialogs:
                    self.process_dlg_dir('train', DatasetOption.train_dialog_data_directory, self.train_dialogs)
                RawData.save_dialogs_to_pkl(self.train_dialogs, DatasetOption.train_pkl)
            else:
                train_item_pkl = getattr(DatasetOption, '{}_train_item_pkl'.format(self.task))
                if not isfile(train_item_pkl):
                    self.train_dialogs = pkl.load(open(DatasetOption.train_pkl, 'rb'))

        # get valid dialogs
        if valid:
            if not isfile(DatasetOption.valid_pkl):
                if not self.valid_dialogs:
                    self.process_dlg_dir('valid', DatasetOption.valid_dialog_data_directory, self.valid_dialogs)
                RawData.save_dialogs_to_pkl(self.valid_dialogs, DatasetOption.valid_pkl)
            else:
                valid_item_pkl = getattr(DatasetOption, '{}_valid_item_pkl'.format(self.task))
                if not isfile(valid_item_pkl):
                    self.valid_dialogs = pkl.load(open(DatasetOption.valid_pkl, 'rb'))

        # get test dialogs
        if test:
            if not isfile(DatasetOption.test_pkl):
                if not self.test_dialogs:
                    self.process_dlg_dir('test', DatasetOption.test_dialog_data_directory, self.test_dialogs)
                RawData.save_dialogs_to_pkl(self.test_dialogs, DatasetOption.test_pkl)
            else:
                test_item_pkl = getattr(DatasetOption, '{}_test_item_pkl'.format(self.task))
                if not isfile(test_item_pkl):
                    self.test_dialogs = pkl.load(open(DatasetOption.test_pkl, 'rb'))

        self.prod_collection = ProductCollection()
        self.prod_vocab = None

        # get product vocab
        if not isfile(DatasetOption.product_vocab_pkl):
            if isfile(DatasetOption.product_pkl):
                raise ValueError("Product vocab doesn't exist but pkl dataset exists.")
            self.get_prod_vocab()
        else:
            self.prod_vocab = pkl.load(open(DatasetOption.product_vocab_pkl, 'rb'))

        # get product data
        if not isfile(DatasetOption.product_pkl):
            if self.prod_collection.empty():
                for prod_dir in DatasetOption.product_data_directories:
                    self.process_prod_dir(prod_dir)
            self.save_prods_to_pkl()
        else:
            self.prod_collection = pkl.load(open(DatasetOption.product_pkl, 'rb'))

        # url2img
        self.url2img = dict([tuple(line.strip().split(' ')) for line in open(DatasetOption.url2img, 'r').readlines()])

        # glove
        print('read glove {}...'.format(DatasetOption.glove_file))
        with open(DatasetOption.glove_file, 'r') as f:
            self.glove = {}
            for line in f:
                line = line.strip().split(' ')
                if line:
                    self.glove[line[0]] = list(map(float, line[1:]))

    def get_vocab(self):
        print('get vocab...')
        word_freq = Counter()
        self.process_dlg_dir('train', DatasetOption.train_dialog_data_directory, self.train_dialogs, word_freq)
        self.process_dlg_dir('valid', DatasetOption.valid_dialog_data_directory, self.valid_dialogs, word_freq)
        words = ['</s>', '</e>', '<unk>', '<pad>'] + \
                [word for word, freq in word_freq.most_common() if freq >= DatasetOption.context_text_cutoff]
        self.vocab = {word: wid for wid, word in enumerate(words)}
        print('save vocab to {}...'.format(DatasetOption.vocab_pkl))
        with open(DatasetOption.vocab_pkl, 'wb') as f:
            pkl.dump(self.vocab, f)
        print('saved')

    @staticmethod
    def save_dialogs_to_pkl(dialogs, pkl_file):
        # save dialogs to pkl_file
        print('save dialog to {}...'.format(pkl_file))
        with open(pkl_file, 'wb') as f:
            pkl.dump(dialogs, f)
        print('get vocab...')

    @staticmethod
    def process_dlg_dir(data_type, dlg_dir, dialogs, word_freq=None):
        print('process dialog dir {}...'.format(dlg_dir))
        files = os.listdir(dlg_dir)
        for file_idx, file in enumerate(files):
            if file.endswith('.json'):
                full_path = os.path.join(dlg_dir, file)
                print('process dialog dir: {}/{}'.format(file_idx + 1, len(files)))
                try:
                    dialog_dict = json.load(open(full_path))
                except json.decoder.JSONDecodeError:
                    continue
                # extract useful info
                dialog = []
                for utter in dialog_dict:
                    # get utter attributes
                    speaker = utter.get('speaker')
                    utter = utter.get('utterance')
                    text = utter.get('nlg')
                    images = utter.get('images')
                    false_images = utter.get('false images')
                    # some attributes may be empty
                    if text is None:
                        text = ""
                    if images is None:
                        images = []
                    if false_images is None:
                        false_images = []
                    # append it to the utters
                    dialog.append(Utterance(speaker, text, images, false_images))
                    # update word_freq if needed
                    if word_freq is not None:
                        word_freq.update([word.lower() for word in word_tokenize(text)])
                dialogs.append(dialog)

    def get_prod_vocab(self):
        print('get product vocab...')
        word_freq = Counter()
        if not isfile(DatasetOption.product_pkl):
            for prod_dir in DatasetOption.product_data_directories:
                self.process_prod_dir(prod_dir, word_freq)
        words = ['</s>', '</e>', '<unk>', '<pad>'] + \
                [word for word, freq in word_freq.most_common() if freq >= DatasetOption.product_text_cutoff]
        self.prod_vocab = {word: wid for wid, word in enumerate(words)}
        print('save product vocab to {}...'.format(DatasetOption.product_vocab_pkl))
        with open(DatasetOption.product_vocab_pkl, 'wb') as f:
            pkl.dump(self.prod_vocab, f)
        print('saved')

    def process_prod_dir(self, prod_dir, word_freq=None):
        print('process product dir {}...'.format(prod_dir))
        files = os.listdir(prod_dir)
        for file_idx, file in enumerate(files):
            if file.endswith('.json'):
                full_path = os.path.join(prod_dir, file)
                print('process product dir: {}/{}'.format(file_idx + 1, len(files)))
                try:
                    json_dict = json.load(open(full_path))
                except json.decoder.JSONDecodeError:
                    continue
                # use json_dict to construct prod
                prod = Product(json_dict)
                # insert the new product to prod_collection
                self.prod_collection.insert(prod)
                # update word_freq if needed
                if word_freq is not None:
                    word_freq.update([word.lower() for word in word_tokenize(prod.prod_str)])

    def save_prods_to_pkl(self):
        print('save products to {}...'.format(DatasetOption.product_pkl))
        with open(DatasetOption.product_pkl, 'wb') as f:
            pkl.dump(self.prod_collection, f)
        print('saved')
