import json
from time import time
from utils import language_utils

import functools
print = functools.partial(print, flush=True)


class CocoDatasetKarpathy:

    TrainSet_ID = 1
    ValidationSet_ID = 2
    TestSet_ID = 3

    def __init__(self,
                 images_path,
                 coco_annotations_path,
                 train2014_bboxes_path,
                 val2014_bboxes_path,
                 precalc_features_hdf5_filepath,
                 preproc_images_hdf5_filepath=None,
                 limited_num_train_images=None,
                 limited_num_val_images=None,
                 limited_num_test_images=None,
                 dict_min_occurrences=5,
                 verbose=True
                 ):
        super(CocoDatasetKarpathy, self).__init__()

        self.use_images_instead_of_features = False
        if precalc_features_hdf5_filepath is None or precalc_features_hdf5_filepath == 'None' or \
                precalc_features_hdf5_filepath == 'none' or precalc_features_hdf5_filepath == '':
            self.use_images_instead_of_features = True
            print("Warning: since no hdf5 path is provided using images instead of pre-calculated features.")
            print("Features path: " + str(precalc_features_hdf5_filepath))

            self.preproc_images_hdf5_filepath = None
            if preproc_images_hdf5_filepath is not None:
                print("Preprocessed hdf5 file path not None: " + str(preproc_images_hdf5_filepath))
                print("Using preprocessed hdf5 file instead.")
                self.preproc_images_hdf5_filepath = preproc_images_hdf5_filepath

        else:
            self.precalc_features_hdf5_filepath = precalc_features_hdf5_filepath
            print("Features path: " + str(self.precalc_features_hdf5_filepath))
            print("BBoxes features path provided, images are provided in form of features.")

        if images_path is None:
            print("Warning: images path set to None, the program will run but some debug method in " + str(__file__) + \
                  " will lead to program termination, as a portion of the image path will be replaced with an empty string")
            self.images_path = ""
        else:
            self.images_path = images_path

        self.karpathy_train_set = dict()
        self.karpathy_val_set = dict()
        self.karpathy_test_set = dict()

        with open(coco_annotations_path, 'r') as f:
            json_file = json.load(f)['images']

        num_train_captions = 0
        num_val_captions = 0
        num_test_captions = 0
        if verbose:
            print("Initializing dataset... ", end=" ")
        for json_item in json_file:
            new_item = dict()

            new_item['img_path'] = self.images_path + json_item['filepath'] + '/img/' + json_item['filename']

            new_item_captions = [item['raw'] for item in json_item['sentences']]
            new_item['img_id'] = json_item['cocoid']
            new_item['captions'] = new_item_captions

            if json_item['split'] == 'train' or json_item['split'] == 'restval':
                self.karpathy_train_set[json_item['cocoid']] = new_item
                num_train_captions += len(json_item['sentences'])
            elif json_item['split'] == 'test':
                self.karpathy_test_set[json_item['cocoid']] = new_item
                num_test_captions += len(json_item['sentences'])
            elif json_item['split'] == 'val':
                self.karpathy_val_set[json_item['cocoid']] = new_item
                num_val_captions += len(json_item['sentences'])

        self.add_bboxes(train2014_bboxes_path)
        self.add_bboxes(val2014_bboxes_path)

        list_train_set = []
        list_val_set = []
        list_test_set = []
        for key in self.karpathy_train_set.keys():
            list_train_set.append(self.karpathy_train_set[key])
        for key in self.karpathy_val_set.keys():
            list_val_set.append(self.karpathy_val_set[key])
        for key in self.karpathy_test_set.keys():
            list_test_set.append(self.karpathy_test_set[key])
        self.karpathy_train_list = list_train_set
        self.karpathy_val_list = list_val_set
        self.karpathy_test_list = list_test_set

        self.train_num_images = len(self.karpathy_train_list)
        self.val_num_images = len(self.karpathy_val_list)
        self.test_num_images = len(self.karpathy_test_list)

        if limited_num_train_images is not None:
            self.karpathy_train_list = self.karpathy_train_list[:limited_num_train_images]
            self.train_num_images = limited_num_train_images
        if limited_num_val_images is not None:
            self.karpathy_val_list = self.karpathy_val_list[:limited_num_val_images]
            self.val_num_images = limited_num_val_images
        if limited_num_test_images is not None:
            self.karpathy_test_list = self.karpathy_test_list[:limited_num_test_images]
            self.test_num_images = limited_num_test_images

        if verbose:
            print("Num train images: " + str(self.train_num_images))
            print("Num val images: " + str(self.val_num_images))
            print("Num test images: " + str(self.test_num_images))

        tokenized_captions_list = []
        for i in range(self.train_num_images):
            for caption in self.karpathy_train_list[i]['captions']:
                tmp = language_utils.lowercase_and_clean_trailing_spaces([caption])
                tmp = language_utils.add_space_between_non_alphanumeric_symbols(tmp)
                tmp = language_utils.remove_punctuations(tmp)
                tokenized_caption = ['SOS'] + language_utils.tokenize(tmp)[0] + ['EOS']
                tokenized_captions_list.append(tokenized_caption)

        counter_dict = dict()
        for i in range(len(tokenized_captions_list)):
            for word in tokenized_captions_list[i]:
                if word not in counter_dict:
                    counter_dict[word] = 1
                else:
                    counter_dict[word] += 1

        less_than_min_occurrences_set = set()
        for k, v in counter_dict.items():
            if v < dict_min_occurrences:
                less_than_min_occurrences_set.add(k)
        if verbose:
            print("tot tokens " + str(len(counter_dict)) +
                  " less than " + str(dict_min_occurrences) + ": " + str(len(less_than_min_occurrences_set)) +
                  " remaining: " + str(len(counter_dict) - len(less_than_min_occurrences_set)))

        self.num_caption_vocab = 4
        self.max_seq_len = 0
        discovered_words = ['PAD', 'SOS', 'EOS', 'UNK']
        for i in range(len(tokenized_captions_list)):
            caption = tokenized_captions_list[i]
            if len(caption) > self.max_seq_len:
                self.max_seq_len = len(caption)
            for word in caption:
                if (word not in discovered_words) and (not word in less_than_min_occurrences_set):
                    discovered_words.append(word)
                    self.num_caption_vocab += 1

        discovered_words.sort()
        self.caption_word2idx_dict = dict()
        self.caption_idx2word_list = []
        for i in range(len(discovered_words)):
            self.caption_word2idx_dict[discovered_words[i]] = i
            self.caption_idx2word_list.append(discovered_words[i])
        if verbose:
            print("There are " + str(self.num_caption_vocab) + " vocabs in dict")

    def add_bboxes(self, annotations_path):
        with open(annotations_path, 'r') as f:
            annotation_dicts = json.load(f)['annotations']
        for entry in annotation_dicts:
            img_id = entry['image_id']
            if img_id in self.karpathy_val_set.keys():
                dict_reference = self.karpathy_val_set
            elif img_id in self.karpathy_test_set.keys():
                dict_reference = self.karpathy_test_set
            else:  # if img_id in train_caption_ids:
                dict_reference = self.karpathy_train_set
            bbox = entry['bbox']
            x, y, weight, height = bbox
            new_format_bbox = (int(x), int(y), int(x + weight), int(y + height))
            if 'bboxes' not in dict_reference[img_id].keys():
                dict_reference[img_id]['bboxes'] = [new_format_bbox]
            else:
                dict_reference[img_id]['bboxes'].append(new_format_bbox)

    def get_image_path(self, img_idx, dataset_split):

        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            img_path = self.karpathy_test_list[img_idx]['img_path']
            img_id = self.karpathy_test_list[img_idx]['img_id']
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            img_path = self.karpathy_val_list[img_idx]['img_path']
            img_id = self.karpathy_val_list[img_idx]['img_id']
        else:
            img_path = self.karpathy_train_list[img_idx]['img_path']
            img_id = self.karpathy_train_list[img_idx]['img_id']

        return img_path, img_id

    def get_all_images_captions(self, dataset_split):
        all_image_references = []

        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            dataset = self.karpathy_test_list
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            dataset = self.karpathy_val_list
        else:
            dataset = self.karpathy_train_list

        for img_idx in range(len(dataset)):
            all_image_references.append(dataset[img_idx]['captions'])
        return all_image_references

    def get_eos_token_idx(self):
        return self.caption_word2idx_dict['EOS']

    def get_sos_token_idx(self):
        return self.caption_word2idx_dict['SOS']

    def get_pad_token_idx(self):
        return self.caption_word2idx_dict['PAD']

    def get_unk_token_idx(self):
        return self.caption_word2idx_dict['UNK']

    def get_eos_token_str(self):
        return 'EOS'

    def get_sos_token_str(self):
        return 'SOS'

    def get_pad_token_str(self):
        return 'PAD'

    def get_unk_token_str(self):
        return 'UNK'