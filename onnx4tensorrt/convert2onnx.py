
import os
import numpy as np
import torch
import subprocess
import torchvision
import argparse
import pickle
import copy
import onnx
from argparse import Namespace

from PIL import Image as PIL_Image

import functools
print = functools.partial(print, flush=True)

from utils.saving_utils import partially_load_state_dict
from utils.language_utils import tokens2description
from onnx4tensorrt.End_ExpansionNet_v2_onnx_tensorrt import End_ExpansionNet_v2_ONNX_TensorRT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conversion PyTorch to ONNX')
    parser.add_argument('--model_dim', type=int, default=512)
    parser.add_argument('--N_enc', type=int, default=3)
    parser.add_argument('--N_dec', type=int, default=3)
    parser.add_argument('--vocab_path', type=str, default='./demo_material/demo_coco_tokens.pickle')
    parser.add_argument('--max_seq_len', type=int, default=74)
    parser.add_argument('--image_path_1', type=str, default='./demo_material/tatin.jpg')
    parser.add_argument('--image_path_2', type=str, default='./demo_material/micheal.jpg')
    parser.add_argument('--load_model_path', type=str, default='./github_ignore_material/saves/rf_model.pth')
    parser.add_argument('--output_onnx_path', type=str, default='./rf_model.onnx')
    parser.add_argument('--onnx_simplify', type=bool, default=True)
    parser.add_argument('--onnx_runtime_test', type=bool, default=True)
    parser.add_argument('--onnx_tensorrt_test', type=bool, default=True)
    parser.add_argument('--max_worker_size', type=int, default=10000)
    parser.add_argument('--onnx_opset', type=int, default=14)
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()

    img_size = 384
    with open(args.vocab_path, 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    # Pre-Processing
    def preprocess_image(image_path):
        transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
        transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])

        pil_image = PIL_Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = PIL_Image.new("RGB", pil_image.size)
        preprocess_pil_image = transf_1(pil_image)
        image = torchvision.transforms.ToTensor()(preprocess_pil_image)
        image = transf_2(image)
        return image.unsqueeze(0)

    # test the generalization of the graph using two images
    image_1 = preprocess_image(args.image_path_1)
    image_2 = preprocess_image(args.image_path_2)

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model = End_ExpansionNet_v2_ONNX_TensorRT(
                swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                swin_norm_layer=torch.nn.LayerNorm, swin_patch_norm=True,

                d_model=args.model_dim, N_enc=args.N_enc,
                N_dec=args.N_dec, num_heads=8, ff=2048,
                num_exp_enc_list=[32, 64, 128, 256, 512],
                num_exp_dec=16,
                output_word2idx=coco_tokens['word2idx_dict'],
                output_idx2word=coco_tokens['idx2word_list'],
                max_seq_len=args.max_seq_len, drop_args=drop_args)

    print("Loading model...")
    checkpoint = torch.load(args.load_model_path)
    partially_load_state_dict(model, checkpoint['model_state_dict'])

    print("===============================================")
    print("||                ONNX conversion            ||")
    print("===============================================")

    print("Exporting...")
    model.eval()
    my_script = torch.jit.script(model)
    torch.onnx.export(
        my_script,
        (image_1, torch.tensor([0]), torch.tensor([sos_idx])),
        args.output_onnx_path,
        input_names=['enc_x', 'enc_x_num_pads', 'sos_idx'],
        output_names=['pred', 'logprobs'],
        export_params=True,
        opset_version=args.onnx_opset)
    print("ONNX graph conversion done. Destination: " + args.output_onnx_path)
    onnx_model = onnx.load(args.output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX graph checked.")

    if args.onnx_simplify:
        print("===============================================")
        print("||       ONNX graph simplifcation phase      ||")
        print("===============================================")

        simplified_onnx_path = args.output_onnx_path + "_simplified"
        popen = subprocess.Popen(["onnxsim " + args.output_onnx_path + " " + simplified_onnx_path],
                                 env=os.environ, cwd='./', stdout=subprocess.PIPE)
        popen_out, popen_err = popen.communicate()
        print("ONNX simplifer output:\n" + str(popen_out))
        print("ONNX simplifier errors:\n" + str(popen_err))
        print("Generated simiplified version of ONNX: " + simplified_onnx_path)
        print("Following tests will be performed on the simplified ONNX: " + simplified_onnx_path)
        onnx_model = onnx.load(simplified_onnx_path)

    if args.onnx_runtime_test:
        import onnxruntime as ort
        print("===============================================")
        print("||      Testing on ONNX Runtime testing      ||")
        print("===============================================")

        ort_sess = ort.InferenceSession(args.output_onnx_path)
        input_dict_1 = {'enc_x': image_1.numpy(),
                        'enc_x_num_pads': np.array([0]),
                        'sos_idx': np.array([sos_idx])}
        outputs_ort = ort_sess.run(None, input_dict_1)
        output_caption = tokens2description(outputs_ort[0][0].tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print("ONNX Runtime result on 1st image:\n\t\t" + output_caption)

        input_dict_2 = copy.copy(input_dict_1)
        input_dict_2['enc_x'] = image_2.numpy()
        outputs_ort = ort_sess.run(None, input_dict_2)
        output_caption = tokens2description(outputs_ort[0][0].tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print("ONNX Runtime result on 2nd image:\n\t\t" + output_caption)
        print("Done.", end="\n\n")

    if args.onnx_tensorrt_test:
        import onnx_tensorrt.backend as backend
        print("===============================================")
        print("||      Testing on ONNX-TensorRT backend     ||")
        print("===============================================")

        engine = backend.prepare(onnx_model, device='CUDA:0', max_worker_size=args.max_worker_size)

        input_data = [image_1.numpy(), np.array([0]), np.array([sos_idx])]
        output_data = engine.run(input_data)[0][0]
        output_caption = tokens2description(output_data.tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print("TensorRT result on 1st image:\n\t\t" + output_caption)

        input_data[0] = image_2.numpy()
        output_data = engine.run(input_data)[0][0]
        output_caption = tokens2description(output_data.tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print("TensorRT result on 2nd image:\n\t\t" + output_caption)
        print("Done.", end="\n\n")

    print("Closing.")

