
import numpy as np
import torch
import argparse
import pickle
import copy
import onnx
from argparse import Namespace


import functools
print = functools.partial(print, flush=True)

from utils.saving_utils import partially_load_state_dict
from utils.image_utils import preprocess_image
from utils.language_utils import tokens2description
from onnx4tensorrt.End_ExpansionNet_v2_onnx_tensorrt import create_pad_mask, create_no_peak_and_pad_mask
from onnx4tensorrt.End_ExpansionNet_v2_onnx_tensorrt import End_ExpansionNet_v2_ONNX_TensorRT, \
    NUM_FEATURES, MAX_DECODE_STEPS


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
    parser.add_argument('--onnx_simplify', type=bool, default=False)
    parser.add_argument('--onnx_runtime_test', type=bool, default=False)
    parser.add_argument('--onnx_tensorrt_test', type=bool, default=False)
    parser.add_argument('--max_worker_size', type=int, default=10000)
    parser.add_argument('--onnx_opset', type=int, default=14)
    # parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()

    with open(args.vocab_path, 'rb') as f:
        coco_tokens = pickle.load(f)
        sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
        eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

    # test the generalization of the graph using two images
    img_size = 384
    image_1 = preprocess_image(args.image_path_1, img_size)
    image_2 = preprocess_image(args.image_path_2, img_size)

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    enc_exp_list = [32, 64, 128, 256, 512]
    dec_exp = 16
    num_heads = 8
    model = End_ExpansionNet_v2_ONNX_TensorRT(
                swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                swin_norm_layer=torch.nn.LayerNorm, swin_patch_norm=True,

                d_model=args.model_dim, N_enc=args.N_enc,
                N_dec=args.N_dec, num_heads=num_heads, ff=2048,
                num_exp_enc_list=enc_exp_list,
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

    # Masks creation - - - - -
    batch_size = 1
    enc_mask = create_pad_mask(mask_size=(batch_size, sum(enc_exp_list), NUM_FEATURES),
                               pad_row=[0], pad_column=[0]).contiguous()
    no_peak_mask = create_no_peak_and_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS),
                                               num_pads=[0]).contiguous()
    cross_mask = create_pad_mask(mask_size=(batch_size, MAX_DECODE_STEPS, NUM_FEATURES),
                                 pad_row=[0], pad_column=[0]).contiguous()
    # contrary to the other masks, we put 1 in correspondence to the values to be masked
    cross_mask = 1 - cross_mask

    fw_dec_mask = no_peak_mask.unsqueeze(2).expand(batch_size, MAX_DECODE_STEPS,
                                                   dec_exp, MAX_DECODE_STEPS).contiguous(). \
                    view(batch_size, MAX_DECODE_STEPS * dec_exp, MAX_DECODE_STEPS)

    bw_dec_mask = no_peak_mask.unsqueeze(-1).expand(batch_size,
            MAX_DECODE_STEPS, MAX_DECODE_STEPS, dec_exp).contiguous(). \
            view(batch_size, MAX_DECODE_STEPS, MAX_DECODE_STEPS * dec_exp)

    atten_mask = cross_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

    # - - - - - - - - - -

    print("Exporting...")
    model.eval()
    my_script = torch.jit.script(model)
    torch.onnx.export(
        my_script,
        (image_1, torch.tensor([sos_idx]), enc_mask, fw_dec_mask, bw_dec_mask, atten_mask),
        args.output_onnx_path,
        input_names=['enc_x', 'sos_idx', 'enc_mask', 'fw_dec_mask', 'bw_dec_mask', 'cross_mask'],
        output_names=['pred', 'logprobs'],
        export_params=True,
        opset_version=args.onnx_opset)
    print("ONNX graph conversion done. Destination: " + args.output_onnx_path)
    onnx_model_fp32 = onnx.load(args.output_onnx_path)
    onnx.checker.check_model(onnx_model_fp32)
    print("ONNX graph checked.")

    # TO-DO: FP16 code was written but does not work for reason yet unknown,
    # tests were made on model trained in FP32
    # what if the model was trained in FP16 instead?

    # from onnxconverter_common import float16
    # onnx_model_fp16 = float16.convert_float_to_float16(onnx_model_fp32, op_block_list=["Topk", "Normalizer"])
    # onnx.save(onnx_model_fp16, args.output_onnx_path + '_fp16')
    # print("ONNX graph FP16 version done. Destination: " + args.output_onnx_path + '_fp16')

    if args.onnx_simplify:
        print("===============================================")
        print("||       ONNX graph simplifcation phase      ||")
        print("===============================================")
        from onnxsim import simplify
        onnx_model_fp32 = onnx.load(args.output_onnx_path)
        # onnx_model_fp16 = onnx.load(args.output_onnx_path + '_fp16')
        try:
            simplified_onnx_model_fp32, check_fp32 = simplify(onnx_model_fp32)
            # simplified_onnx_model_fp16, check_fp16 = simplify(onnx_model_fp16)
        except:
            print("The simplification failed. In this case, we suggest to try the command line version.")
        assert check_fp32, "Simplified fp32 ONNX model could not be validated"
        # assert check_fp16, "Simplified fp16 ONNX model could not be validated"
        onnx.save(simplified_onnx_model_fp32, args.output_onnx_path)
        # onnx.save(simplified_onnx_model_fp16, args.output_onnx_path + '_fp16')

    if True:  #args.onnx_runtime_test:
        import onnxruntime as ort
        print("===============================================")
        print("||           Testing on ONNX Runtime         ||")
        print("===============================================")

        ort_sess = ort.InferenceSession(args.output_onnx_path)
        input_dict_1 = {'enc_x': image_1.numpy(), 'sos_idx': np.array([sos_idx]),
                        'enc_mask': enc_mask.numpy(), 'fw_dec_mask': fw_dec_mask.numpy(),
                        'bw_dec_mask': bw_dec_mask.numpy(), 'cross_mask': atten_mask.numpy()}
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

        input_data = [image_1.numpy(), np.array([sos_idx]),
                        enc_mask.numpy(), fw_dec_mask.numpy(), bw_dec_mask.numpy(), atten_mask.numpy()]
        output_data = engine.run(input_data)[0][0]
        output_caption = tokens2description(output_data.tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print("TensorRT result on 1st image:\n\t\t" + output_caption)

        input_data[0] = image_2.numpy()
        output_data = engine.run(input_data)[0][0]
        output_caption = tokens2description(output_data.tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        print("TensorRT result on 2nd image:\n\t\t" + output_caption)
        print("Done.", end="\n\n")

    print("Closing.")

