import h5py
import numpy as np
from PIL import Image as PIL_Image
import torchvision
import torch
import argparse
from argparse import Namespace
from torch.nn.parameter import Parameter
from time import time

from data.coco_dataset import CocoDatasetKarpathy

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import functools
print = functools.partial(print, flush=True)
DEFAULT_RANK = 0


def convert_time_as_hhmmss(ticks):
    return str(int(ticks / 60)) + " m " + \
           str(int(ticks) % 60) + " s"


def generate_data(path_args):

    coco_dataset = CocoDatasetKarpathy(images_path=path_args.image_path,
                                       coco_annotations_path=args.captions_path + "dataset_coco.json",
                                       train2014_bboxes_path=args.captions_path + "train2014_instances.json",
                                       val2014_bboxes_path=args.captions_path + "val2014_instances.json",
                                       preproc_images_hdf5_filepath=None,
                                       precalc_features_hdf5_filepath=None,
                                       limited_num_train_images=None,
                                       limited_num_val_images=5000)

    from models.swin_transformer_mod import SwinTransformer
    model = SwinTransformer(
            img_size=384, patch_size=4, in_chans=3,
            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
            window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0.0, attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False)

    def load_backbone_only_from_save(model, state_dict, prefix=False):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if prefix:
                name = name.lstrip('swin_transf.')
            if name not in own_state:
                print("Not found: " + str(name))
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)
            print("Found: " + str(name))

    save_model_path = path_args.save_path
    map_location = {'cuda:%d' % DEFAULT_RANK: 'cuda:%d' % DEFAULT_RANK}
    checkpoint = torch.load(save_model_path, map_location=map_location)
    if 'model_state_dict' in checkpoint.keys():
        print("Custom save point found")
        load_backbone_only_from_save(model, checkpoint['model_state_dict'], prefix=True)
    else:
        print("Custom save point not found")
        load_backbone_only_from_save(model, checkpoint['model'], prefix=False)
    print("Loading phase ended")

    model = model.to(DEFAULT_RANK)

    test_preprocess_layers_1 = [torchvision.transforms.Resize((384, 384))]
    test_preprocess_layers_2 = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    test_preprocess_1 = torchvision.transforms.Compose(test_preprocess_layers_1)
    test_preprocess_2 = torchvision.transforms.Compose(test_preprocess_layers_2)

    model.eval()
    with torch.no_grad():

        hdf5_file = h5py.File(path_args.output_path, 'w')

        def apply_model(model, file_path):
            pil_image = PIL_Image.open(file_path)
            if pil_image.mode != 'RGB':
                pil_image = PIL_Image.new("RGB", pil_image.size)
            preprocess_pil_image = test_preprocess_1(pil_image)
            tens_image = torchvision.transforms.ToTensor()(preprocess_pil_image)
            tens_image = test_preprocess_2(tens_image).to(DEFAULT_RANK)
            output = model(tens_image.unsqueeze(0))
            return output.squeeze(0)

        for i in range(coco_dataset.train_num_images):
           img_path, img_id = coco_dataset.get_image_path(coco_dataset.train_num_images - i - 1,
                                                          CocoDatasetKarpathy.TrainSet_ID)
           output = apply_model(model, img_path)
           hdf5_file.create_dataset(str(img_id) + '_features', data=np.array(output.cpu()))
           if i % 20000 == 0 or i == coco_dataset.train_num_images - 1:
               print("Train " + str(i) + " / " + str(coco_dataset.train_num_images) + " completed")

        for i in range(coco_dataset.test_num_images):
            img_path, img_id = coco_dataset.get_image_path(i, CocoDatasetKarpathy.TestSet_ID)
            output = apply_model(model, img_path)
            hdf5_file.create_dataset(str(img_id) + '_features', data=np.array(output.cpu()))
            if i % 2500 == 0 or i == coco_dataset.test_num_images - 1:
                print("Test " + str(i) + " / " + str(coco_dataset.test_num_images) + " completed")

        for i in range(coco_dataset.val_num_images):
            img_path, img_id = coco_dataset.get_image_path(i, CocoDatasetKarpathy.ValidationSet_ID)
            output = apply_model(model, img_path)
            hdf5_file.create_dataset(str(img_id) + '_features', data=np.array(output.cpu()))
            if i % 2500 == 0 or i == coco_dataset.test_num_images - 1:
                print("Val " + str(i) + " / " + str(coco_dataset.test_num_images) + " completed")

    print("[GPU: " + str(DEFAULT_RANK) + " ] Closing...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--save_path', type=str, default='./github_ignore_material/saves/')
    parser.add_argument('--output_path', type=str, default='./github_ignore_material/raw_data/precalc_features.hdf5')
    parser.add_argument('--image_path', type=str, default='/tmp/images/')
    parser.add_argument('--captions_path', type=str, default='./github_ignore_material/raw_data/')

    args = parser.parse_args()

    path_args = Namespace(save_path=args.save_path,
                          output_path=args.output_path,
                          image_path=args.image_path,
                          captions_path=args.captions_path)
    generate_data(path_args=path_args)


