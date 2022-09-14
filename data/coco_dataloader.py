import numpy as np
import random
import h5py
import copy
import torch
from time import time
from data.coco_dataset import CocoDatasetKarpathy

from utils import language_utils
from data.transparent_data_loader import TransparentDataLoader


from PIL import Image as PIL_Image
import torchvision


import functools
print = functools.partial(print, flush=True)


class CocoDataLoader(TransparentDataLoader):

    NOT_DEFINED = -1

    def __init__(self, coco_dataset,
                       array_of_init_seeds,
                       batch_size, rank=0, num_procs=1,
                       dataloader_mode='caption_wise',
                       resize_image_size=None,
                       verbose=False):
        super(TransparentDataLoader, self).__init__()
        assert (dataloader_mode == 'caption_wise' or dataloader_mode == 'image_wise'), \
            "dataloader_mode must be either caption_wise or image_wise"

        self.coco_dataset = coco_dataset

        self.dataloader_mode = dataloader_mode

        self.num_procs = num_procs
        self.rank = rank

        self.epoch_it = 0
        self.array_of_init_seeds = array_of_init_seeds * 10
        self.max_num_epoch = len(array_of_init_seeds)

        self.max_num_regions = None

        self.batch_size = batch_size

        self.num_procs = num_procs
        self.num_batches = CocoDataLoader.NOT_DEFINED
        self.batch_it = []
        self.image_idx_x = []
        self.caption_y = []
        for idx_proc in range(num_procs):
            self.batch_it.append(0)
            self.image_idx_x.append([])
            self.caption_y.append([])

        self.use_images_instead_of_features = False
        if self.coco_dataset.use_images_instead_of_features:
            print("Warning: using Images instead of features in the DataLoader")

            if self.coco_dataset.preproc_images_hdf5_filepath is not None:
                self.hdf5_img_file = h5py.File(self.coco_dataset.preproc_images_hdf5_filepath, 'r', rdcc_nbytes=0)

            self.use_images_instead_of_features = True
            assert (resize_image_size is not None), "resize_image_size must be defined"
            preprocess_layers_1 = [torchvision.transforms.Resize((resize_image_size, resize_image_size))]
            preprocess_layers_2 = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

            self.train_preprocess_1 = torchvision.transforms.Compose(preprocess_layers_1)
            self.train_preprocess_2 = torchvision.transforms.Compose(preprocess_layers_2)

            self.test_preprocess_1 = torchvision.transforms.Compose(preprocess_layers_1)
            self.test_preprocess_2 = torchvision.transforms.Compose(preprocess_layers_2)

            self.debug_counter = 0

        else:
            self.hdf5_file = h5py.File(self.coco_dataset.precalc_features_hdf5_filepath, 'r', rdcc_nbytes=0)

        self.set_epoch_it(epoch=0, verbose=verbose)

    def init_epoch(self, epoch_it, verbose=False):
        init_timer_start = time()

        batch_size = self.batch_size
        random.seed(self.array_of_init_seeds[epoch_it])

        if self.dataloader_mode == 'caption_wise':
            self.batch_it = []
            self.image_idx_x = []
            self.caption_y = []
            for idx_proc in range(self.num_procs):
                self.batch_it.append(0)
                self.image_idx_x.append([])
                self.caption_y.append([])

            img_idx_caption_id_pair_list = []
            for img_idx in range(self.coco_dataset.train_num_images):
                num_captions = len(self.coco_dataset.karpathy_train_list[img_idx]['captions'])
                for caption_id in range(num_captions):
                    img_idx_caption_id_pair_list.append((img_idx, caption_id))
            random.shuffle(img_idx_caption_id_pair_list)

            tailing_elements = len(img_idx_caption_id_pair_list) % (batch_size * self.num_procs)
            if tailing_elements != 0:
                img_idx_caption_id_pair_list = img_idx_caption_id_pair_list[:-tailing_elements]

            image_idx_batch = []
            caption_y_batch = []
            for idx_proc in range(self.num_procs):
                image_idx_batch.append([])
                caption_y_batch.append([])
            i = 0
            while i < len(img_idx_caption_id_pair_list):
                for idx_proc in range(self.num_procs):
                    img_idx, caption_id = img_idx_caption_id_pair_list[i]
                    image_idx_batch[idx_proc].append(img_idx)
                    preprocessed_caption = self.preprocess(self.coco_dataset.karpathy_train_list[img_idx]['captions'][caption_id])
                    caption_y_batch[idx_proc].append(preprocessed_caption)
                    i += 1
                if i % batch_size == 0:
                    for idx_proc in range(self.num_procs):
                        self.image_idx_x[idx_proc].append(image_idx_batch[idx_proc])
                        self.caption_y[idx_proc].append(caption_y_batch[idx_proc])
                        image_idx_batch[idx_proc] = []
                        caption_y_batch[idx_proc] = []

            self.num_batches = len(self.image_idx_x[0])

            for idx_proc in range(self.num_procs):
                self.batch_it[idx_proc] = 0
        else:  # image_wise
            self.batch_it = []
            self.image_idx_x = []
            for idx_proc in range(self.num_procs):
                self.batch_it.append(0)
                self.image_idx_x.append([])

            img_idxes_list = list(range(self.coco_dataset.train_num_images))
            random.shuffle(img_idxes_list)

            tailing_elements = len(img_idxes_list) % (batch_size * self.num_procs)
            if tailing_elements != 0:
                img_idxes_list = img_idxes_list[:-tailing_elements]

            image_idx_batch = []
            for idx_proc in range(self.num_procs):
                image_idx_batch.append([])
            i = 0
            while i < len(img_idxes_list):
                for idx_proc in range(self.num_procs):
                    img_idx = img_idxes_list[i]
                    image_idx_batch[idx_proc].append(img_idx)
                    i += 1
                if i % batch_size == 0:
                    for idx_proc in range(self.num_procs):
                        self.image_idx_x[idx_proc].append(image_idx_batch[idx_proc])
                        image_idx_batch[idx_proc] = []
            self.num_batches = len(self.image_idx_x[0])
            for idx_proc in range(self.num_procs):
                self.batch_it[idx_proc] = 0

        if verbose:
            print(str(self.rank) + "] " + __name__ + ") Dataset epoch initialization " + str(
                time() - init_timer_start) + " s elapsed")
            print(str(self.rank) + "] " + __name__ + ") How many batches " + str(self.num_batches))

    def add_pad_according_to_batch(self, batch_sentences, pad_symbol):
        batch_size = len(batch_sentences)
        list_of_lengthes = [len(batch_sentences[batch_idx]) for batch_idx in range(batch_size)]
        in_batch_max_seq_len = max(list_of_lengthes)
        batch_num_pads = []
        new_batch_sentences = []
        for batch_idx in range(batch_size):
            num_pads = in_batch_max_seq_len - len(batch_sentences[batch_idx])
            new_batch_sentences.append(batch_sentences[batch_idx] \
                                       + [pad_symbol] * (num_pads))
            batch_num_pads.append(num_pads)
        return new_batch_sentences, batch_num_pads

    def get_next_batch(self, verbose=False, get_also_image_idxes=False, get_also_image_path=False):

        if self.batch_it[self.rank] >= self.num_batches:
            if verbose:
                print("Proc: " + str(self.rank) + " re-initialization")
            self.epoch_it += 1
            if self.epoch_it >= len(self.array_of_init_seeds):
                raise Exception("Please increase number of random seed in the array of initialization seed.")

            self.init_epoch(epoch_it=self.epoch_it, verbose=verbose)

        img_id_batch = []
        img_idx_batch = self.image_idx_x[self.rank][self.batch_it[self.rank]]
        for i in range(len(img_idx_batch)):
            img_idx = img_idx_batch[i]
            img_id_batch.append(self.coco_dataset.karpathy_train_list[img_idx]['img_id'])

        if self.use_images_instead_of_features:
            batch_x, batch_x_num_pads = self.get_PADDED_image_batch_by_idx(img_idx_batch)
        else:
            batch_x, batch_x_num_pads = self.get_PADDED_bboxes_batch_by_id(img_id_batch)

        if self.dataloader_mode == 'caption_wise':
            batch_caption_y_as_string = copy.copy(self.caption_y[self.rank][self.batch_it[self.rank]])
            batch_caption_y_encoded = language_utils. \
                convert_allsentences_word2idx(batch_caption_y_as_string,
                                              self.coco_dataset.caption_word2idx_dict)
            batch_y, batch_y_num_pads = self.add_pad_according_to_batch(batch_caption_y_encoded,
                                                                        self.coco_dataset.caption_word2idx_dict['PAD'])
            batch_y = torch.tensor(batch_y)
        else:  # image_wise
            batch_y = [self.coco_dataset.karpathy_train_list[img_idx]['captions'] for img_idx in img_idx_batch]

        if verbose:
            if not self.use_images_instead_of_features:
                mean_src_len = int(
                    sum([(len(batch_x[i]) - batch_x_num_pads[i]) for i in range(len(batch_x))]) / len(batch_x))
            else:
                mean_src_len = 'Constant'

            if self.dataloader_mode == 'caption_wise':
                mean_trg_len = int(
                    sum([(len(batch_y[i]) - batch_y_num_pads[i]) for i in range(len(batch_y))]) / len(batch_y))
            else:  # image_wise
                mean_trg_len = \
                    sum([(len(cap.split(' '))) for captions in batch_y for cap in captions]) // sum(
                        [len(captions) for captions in batch_y])

            print(str(self.rank) + "] " + __name__ + ") batch " + str(self.batch_it[self.rank]) + " / " +
                  str(self.num_batches) + " batch_size: " + str(len(batch_x)) + " epoch: " + str(self.epoch_it) +
                  " avg_src_seq_len: " + str(mean_src_len) +
                  " avg_trg_seq_len: " + str(mean_trg_len))

        self.batch_it[self.rank] += 1

        file_path_batch_x = []
        if get_also_image_path:
            for i in range(len(img_idx_batch)):
                idx = img_idx_batch[i]
                file_path_batch_x.append(self.coco_dataset.karpathy_train_list[idx]['img_path'])

            if self.dataloader_mode == 'caption_wise':
                return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads, file_path_batch_x
            else:
                return batch_x, batch_y, batch_x_num_pads, file_path_batch_x

        if get_also_image_idxes:
            if self.dataloader_mode == 'caption_wise':
                return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads, img_idx_batch
            else:
                return batch_x, batch_y, batch_x_num_pads, img_idx_batch

        if self.dataloader_mode == 'caption_wise':
            return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads
        else:
            return batch_x, batch_y, batch_x_num_pads

    def get_batch_samples(self, dataset_split, img_idx_batch_list):
        batch_captions_y_as_string = []
        img_id_batch = []
        for i in range(len(img_idx_batch_list)):
            img_idx = img_idx_batch_list[i]

            if dataset_split == CocoDatasetKarpathy.TestSet_ID:
                caption_id = random.randint(a=0, b=len(self.coco_dataset.karpathy_test_list[img_idx]['captions'])-1)
                caption = self.coco_dataset.karpathy_test_list[img_idx]['captions'][caption_id]
            elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
                caption_id = random.randint(a=0, b=len(self.coco_dataset.karpathy_val_list[img_idx]['captions'])-1)
                caption = self.coco_dataset.karpathy_val_list[img_idx]['captions'][caption_id]
            else:
                caption_id = random.randint(a=0, b=len(self.coco_dataset.karpathy_train_list[img_idx]['captions'])-1)
                caption = self.coco_dataset.karpathy_train_list[img_idx]['captions'][caption_id]

            preprocessed_caption = self.preprocess(caption)

            if dataset_split == CocoDatasetKarpathy.TestSet_ID:
               batch_captions_y_as_string.append(preprocessed_caption)
               img_id_batch.append(self.coco_dataset.karpathy_test_list[img_idx]['img_id'])
            elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
               batch_captions_y_as_string.append(preprocessed_caption)
               img_id_batch.append(self.coco_dataset.karpathy_val_list[img_idx]['img_id'])
            else:
               batch_captions_y_as_string.append(preprocessed_caption)
               img_id_batch.append(self.coco_dataset.karpathy_train_list[img_idx]['img_id'])

        if self.use_images_instead_of_features:
            batch_x, batch_x_num_pads = self.get_PADDED_image_batch_by_idx(img_idx_batch_list)
        else:
            batch_x, batch_x_num_pads = self.get_PADDED_bboxes_batch_by_id(img_id_batch)

        batch_caption_y_encoded = language_utils. \
            convert_allsentences_word2idx(batch_captions_y_as_string,
                                          self.coco_dataset.caption_word2idx_dict)
        batch_y, batch_y_num_pads = self.add_pad_according_to_batch(batch_caption_y_encoded,
                                                                    self.coco_dataset.get_pad_token_idx())
        batch_y = torch.tensor(batch_y)

        return batch_x, batch_y, batch_x_num_pads, batch_y_num_pads


    def get_PADDED_image_batch_by_idx(self, img_idx_list, verbose=False):

        list_of_images = []
        for img_idx in img_idx_list:
            if self.coco_dataset.preproc_images_hdf5_filepath is not None:

                img_id = self.coco_dataset.karpathy_train_list[img_idx]['img_id']
                np_array = self.hdf5_img_file[str(img_id) + '_img'][()]
                list_of_images.append(torch.tensor(np_array, dtype=torch.float32).unsqueeze(0))
            else:
                file_path = self.coco_dataset.karpathy_train_list[img_idx]['img_path']

                pil_image = PIL_Image.open(file_path)
                if pil_image.mode != 'RGB':
                    pil_image = PIL_Image.new("RGB", pil_image.size)
                preprocess_pil_image = self.train_preprocess_1(pil_image)
                tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
                tens_image_2 = self.train_preprocess_2(tens_image_1)

                list_of_images.append(tens_image_2.unsqueeze(0))

        self.debug_counter += 1

        return torch.cat(list_of_images), None

    def get_PADDED_bboxes_batch_by_id(self, img_id_list, verbose=False):

        torch.cuda.synchronize()
        start_time = time()

        verbose = False

        list_of_bboxes_tensor = []
        list_of_num_bboxes = []
        for img_id in img_id_list:
            bboxes_numpy_tensor = self.hdf5_file['%d_features' % img_id][()]
            bboxes_tensor = torch.tensor(bboxes_numpy_tensor)
            list_of_bboxes_tensor.append(bboxes_tensor)
            list_of_num_bboxes.append(len(bboxes_numpy_tensor))

        if verbose:
            torch.cuda.synchronize()
            time_spent_batching = (time() - start_time)
            print("Time spent disk I/O: " + str(time_spent_batching) + " s")
            start_time = time()

        output_batch = torch.stack(list_of_bboxes_tensor, dim=0).to(self.rank)

        if verbose:
            time_spent_batching = (time() - start_time)
            print("Time spent memcpy: " + str(time_spent_batching) + " s")
            start_time = time()

        list_of_num_pads = []
        max_seq_len = max([length for length in list_of_num_bboxes])
        for i in range(len(list_of_num_bboxes)):
            list_of_num_pads.append(max_seq_len - list_of_num_bboxes[i])

        if sum(list_of_num_pads) != 0:
            padded_batch_of_bboxes_tensor = torch.nn.utils.rnn.pad_sequence(list_of_bboxes_tensor, batch_first=True)
            output_batch = padded_batch_of_bboxes_tensor
        if verbose:
            time_spent_batching = (time() - start_time)
            print("Time spent batching: " + str(time_spent_batching) + " s")

        return output_batch, list_of_num_pads

    def get_images_by_idx(self, img_idx, dataset_split, transf_mode='train', get_also_id=False):
        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            file_path = self.coco_dataset.karpathy_test_list[img_idx]['img_path']
            img_id = self.coco_dataset.karpathy_test_list[img_idx]['img_id']
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            file_path = self.coco_dataset.karpathy_val_list[img_idx]['img_path']
            img_id = self.coco_dataset.karpathy_val_list[img_idx]['img_id']
        else:
            file_path = self.coco_dataset.karpathy_train_list[img_idx]['img_path']
            img_id = self.coco_dataset.karpathy_train_list[img_idx]['img_id']

        if self.coco_dataset.preproc_images_hdf5_filepath is not None:
            np_array = self.hdf5_img_file[str(img_id) + '_img'][()]
            tens_image = (torch.tensor(np_array, dtype=torch.float32))
        else:
            pil_image = PIL_Image.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = PIL_Image.new("RGB", pil_image.size)
            if transf_mode == 'train':
                pil_image = self.train_preprocess_1(pil_image)
                tens_image = torchvision.transforms.ToTensor()(pil_image)
                tens_image = self.train_preprocess_2(tens_image)
            else:
                pil_image = self.test_preprocess_1(pil_image)
                tens_image = torchvision.transforms.ToTensor()(pil_image)
                tens_image = self.test_preprocess_2(tens_image)

        if get_also_id:
            return tens_image, img_id

        return tens_image

    def get_bboxes_by_idx(self, img_idx, dataset_split):
        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            img_id = self.coco_dataset.karpathy_test_list[img_idx]['img_id']
            bboxes_tensor = torch.tensor(self.hdf5_file['%d_features' % img_id][()])
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            img_id = self.coco_dataset.karpathy_val_list[img_idx]['img_id']
            bboxes_tensor = torch.tensor(self.hdf5_file['%d_features' % img_id][()])
        else:
            img_id = self.coco_dataset.karpathy_train_list[img_idx]['img_id']
            bboxes_tensor = torch.tensor(self.hdf5_file['%d_features' % img_id][()])
        return bboxes_tensor

    def get_all_image_captions_by_idx(self, img_idx, dataset_split):
        if dataset_split == CocoDatasetKarpathy.TestSet_ID:
            caption_list = self.coco_dataset.karpathy_test_list[img_idx]['captions']
        elif dataset_split == CocoDatasetKarpathy.ValidationSet_ID:
            caption_list = self.coco_dataset.karpathy_val_list[img_idx]['captions']
        else:
            caption_list = self.coco_dataset.karpathy_train_list[img_idx]['captions']

        return caption_list

    def get_bboxes_labels(self, img_id_list):
        if self.use_images_instead_of_features:
            pass
        else:
            detected_classes = []
            for i in range(len(img_id_list)):
                img_id = img_id_list[i]
                detected_class = self.hdf5_file['%d_cls_prob' % img_id][()]
                detected_classes.append(list(np.argmax(detected_class, axis=1)))

            return detected_classes

    def set_epoch_it(self, epoch, verbose=False):
        assert (epoch < len(self.array_of_init_seeds)), "requested epoch higher than the maximum: " + str(len(self.array_of_init_seeds))
        self.epoch_it = epoch
        self.init_epoch(epoch_it=self.epoch_it, verbose=verbose)

    def get_epoch_it(self):
        return self.epoch_it

    def get_num_epoch(self):
        return self.max_num_epoch

    def get_num_batches(self):
        return self.num_batches

    def set_batch_it(self, batch_it):
        self.batch_it[self.rank] = batch_it

    def get_batch_it(self):
        return self.batch_it[self.rank]

    def change_batch_size(self, batch_size, verbose):
        self.batch_size = batch_size
        self.set_epoch_it(epoch=0, verbose=verbose)
        self.set_batch_it(batch_it=0)

    def get_batch_size(self):
        return self.batch_size

    def save_state(self):
        return {'batch_it': self.batch_it[self.rank],
                'epoch_it': self.epoch_it,
                'batch_size': self.batch_size,
                'array_of_init_seed': self.array_of_init_seeds}

    def load_state(self, state):
        self.array_of_init_seeds = state['array_of_init_seed']
        self.batch_size = state['batch_size']
        self.set_epoch_it(state['epoch_it'])
        self.batch_it[self.rank] = state['batch_it']

    def preprocess(self, caption):
        caption = language_utils.lowercase_and_clean_trailing_spaces([caption])
        caption = language_utils.add_space_between_non_alphanumeric_symbols(caption)
        caption = language_utils.remove_punctuations(caption)
        caption = [self.coco_dataset.get_sos_token_str()] + language_utils.tokenize(caption)[0] + [self.coco_dataset.get_eos_token_str()]
        preprocessed_tokenized_caption = []
        for word in caption:
            if word not in self.coco_dataset.caption_word2idx_dict.keys():
                preprocessed_tokenized_caption.append(self.coco_dataset.get_unk_token_str())
            else:
                preprocessed_tokenized_caption.append(word)
        return preprocessed_tokenized_caption

    def preprocess_list(self, caption_list):
        for i in range(len(caption_list)):
            caption_list[i] = self.preprocess(caption_list[i])
        return caption_list
