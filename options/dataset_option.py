from os.path import join

from torchvision import transforms


class DatasetOption:
    # ========== PLEASE SET YOUR LOCAL DATA DIRECTORY HERE!!! ==========

    dialog_data_root_directory = '/path/to/dialogs/'
    product_data_root_directory = '/path/to/products/'
    image_root_directory = '/path/to/images/'
    dump_root_directory = '/path/to/dump/'

    # ========== Raw Data ==========

    # dialog data
    train_dialog_data_directory = join(dialog_data_root_directory, "train")
    valid_dialog_data_directory = join(dialog_data_root_directory, "valid")
    test_dialog_data_directory = join(dialog_data_root_directory, "test")

    # product data
    product_data_directories = [join(product_data_root_directory, "com_amazon", "public_jsons"),
                                join(product_data_root_directory, "IN_amazon", "public_jsons"),
                                join(product_data_root_directory, "IN_abof", "public_jsons"),
                                join(product_data_root_directory, "IN_jabong", "public_jsons")]

    # a map from url to image file name
    url2img = 'data/url2img.txt'

    # extracted GloVe vectors, from: http://nlp.stanford.edu/data/glove.840B.300d.zip
    glove_file = 'data/glove.txt'

    # raw dialog data
    train_pkl = join(dump_root_directory, "train_data.pkl")
    valid_pkl = join(dump_root_directory, "valid_data.pkl")
    test_pkl = join(dump_root_directory, "test_data.pkl")

    # vocab for train and valid dataset
    vocab_pkl = join(dump_root_directory, "vocab.pkl")

    # raw product data
    product_pkl = join(dump_root_directory, "product_data.pkl")
    product_vocab_pkl = join(dump_root_directory, "product_vocab.pkl")

    # ========== Dataset ==========

    # number of data loader workers
    num_data_loader_workers = 4

    # extracted item data for text task
    text_train_item_pkl = join(dump_root_directory, "text_train_item.pkl")
    text_valid_item_pkl = join(dump_root_directory, "text_valid_item.pkl")
    text_test_item_pkl = join(dump_root_directory, "text_test_item.pkl")

    # extracted item data for image task
    image_train_item_pkl = join(dump_root_directory, "image_train_item.pkl")
    image_valid_item_pkl = join(dump_root_directory, "image_valid_item.pkl")
    image_test_item_pkl = join(dump_root_directory, "image_test_item.pkl")

    # attributes excluded in product text description
    product_exclude_attributes = ["url", "similar_items"]
    # attributes excluded in product text description but as Product class attributes
    product_attributes = ["image_filename", "image_filename_all", "image_url"]

    # number of utterance
    context_size = 2

    # cutoff
    context_text_cutoff = 4
    last_word_cutoff = 4
    product_text_cutoff = 20

    # max text length
    context_text_length = 30
    product_text_length = 30

    # image
    num_pos_images = 1
    num_neg_images = 4

    image_size = 64

    # image transform
    transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
