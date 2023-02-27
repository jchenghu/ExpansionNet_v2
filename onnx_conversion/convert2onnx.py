"""
    Convert pth model 2 onnx format
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import argparse
import pickle
import onnx
import onnxruntime as ort
import onnx_tensorrt.backend as backend
from argparse import Namespace

from PIL import Image as PIL_Image

import functools
print = functools.partial(print, flush=True)

from utils.saving_utils import partially_load_state_dict
from models.layers import EmbeddingLayer, DecoderLayer, EncoderLayer
from onnx_conversion.swin_transformer_onnx import SwinTransformer_ONNX


# temporary code for ExpansionNet adaptation for ONNX

def create_pad_mask(mask_size, pad_row, pad_column):
    bs, out_len, in_len = mask_size.shape
    pad_row_tens = out_len - pad_row.unsqueeze(-1).repeat(1, out_len)
    pad_col_tens = in_len - pad_column.unsqueeze(-1).repeat(1, in_len)
    arange_on_columns = (torch.arange(in_len).unsqueeze(0).repeat(bs, 1) < pad_col_tens).type(dtype=torch.int32)
    arange_on_rows = (torch.arange(out_len).unsqueeze(0).repeat(bs, 1) < pad_row_tens).type(dtype=torch.int32)
    mask = torch.matmul(arange_on_rows.unsqueeze(-1), arange_on_columns.unsqueeze(-2))
    return mask


def create_no_peak_and_pad_mask(mask_size, num_pads):
    block_mask = create_pad_mask(mask_size, num_pads, num_pads)
    bs, seq_len, seq_len = mask_size.shape
    triang_mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.float),
                             diagonal=0).unsqueeze(0).repeat(bs, 1, 1)
    return torch.mul(block_mask, triang_mask)


class End_ExpansionNet_v2_ONNX(nn.Module):

    def __init__(self,
                 # swin-transf
                 swin_img_size, swin_patch_size, swin_in_chans,
                 swin_embed_dim, swin_depths, swin_num_heads,
                 swin_window_size, swin_mlp_ratio, swin_qkv_bias, swin_qk_scale,
                 swin_drop_rate, swin_attn_drop_rate, swin_drop_path_rate,
                 swin_norm_layer, swin_patch_norm,

                 # captioning
                 d_model, N_enc, N_dec, ff, num_heads, num_exp_enc_list, num_exp_dec,
                 output_word2idx, output_idx2word, max_seq_len, drop_args, rank=0):
        super(End_ExpansionNet_v2_ONNX, self).__init__()

        # swin
        self.swin_transf = SwinTransformer_ONNX(
            img_size=swin_img_size, patch_size=swin_patch_size, in_chans=swin_in_chans,
            embed_dim=swin_embed_dim, depths=swin_depths, num_heads=swin_num_heads,
            window_size=swin_window_size, mlp_ratio=swin_mlp_ratio, qkv_bias=swin_qkv_bias, qk_scale=swin_qk_scale,
            drop_rate=swin_drop_rate, attn_drop_rate=swin_attn_drop_rate, drop_path_rate=swin_drop_path_rate,
            norm_layer=swin_norm_layer, patch_norm=swin_patch_norm)

        self.output_word2idx = output_word2idx
        self.output_idx2word = output_idx2word
        self.max_seq_len = max_seq_len

        self.num_exp_dec = num_exp_dec
        self.num_exp_enc_list = num_exp_enc_list

        self.N_enc = N_enc
        self.N_dec = N_dec
        self.d_model = d_model

        self.encoders = nn.ModuleList(
            [EncoderLayer(d_model, ff, num_exp_enc_list, drop_args.enc) for _ in range(N_enc)])
        self.decoders = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, ff, num_exp_dec, drop_args.dec) for _ in range(N_dec)])

        self.input_embedder_dropout = nn.Dropout(drop_args.enc_input)
        self.input_linear = torch.nn.Linear(1536, d_model)
        self.vocab_linear = torch.nn.Linear(d_model, len(output_word2idx))
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.out_enc_dropout = nn.Dropout(drop_args.other)
        self.out_dec_dropout = nn.Dropout(drop_args.other)

        self.out_embedder = EmbeddingLayer(len(output_word2idx), d_model, drop_args.dec_input)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)

        self.enc_reduce_group = nn.Linear(d_model * self.N_enc, d_model)
        self.enc_reduce_norm = nn.LayerNorm(d_model)
        self.dec_reduce_group = nn.Linear(d_model * self.N_dec, d_model)
        self.dec_reduce_norm = nn.LayerNorm(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.trained_steps = 0
        self.rank = rank

    def forward_enc(self, enc_input: torch.Tensor, enc_input_num_pads: torch.Tensor):

        x = self.swin_transf(enc_input)
        enc_input = self.input_embedder_dropout(self.input_linear(x))
        x = enc_input
        max_num_enc = sum(self.num_exp_enc_list)
        pos_x = torch.arange(max_num_enc).unsqueeze(0).expand(enc_input.size(0), max_num_enc)
        pad_mask = create_pad_mask(mask_size=torch.zeros(enc_input.size(0), max_num_enc, enc_input.size(1)),
                                   pad_row=enc_input_num_pads,
                                   pad_column=enc_input_num_pads)

        x_list = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x=x, n_indexes=pos_x, mask=pad_mask)
            x_list.append(x)
        x_list = torch.cat(x_list, dim=-1)
        x = x + self.out_enc_dropout(self.enc_reduce_group(x_list))
        x = self.enc_reduce_norm(x)

        return x

    def forward_dec(self, cross_input: torch.Tensor,
                    enc_input_num_pads: torch.Tensor, dec_input: torch.Tensor,
                    dec_input_num_pads: torch.Tensor):

        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
            mask_size=torch.zeros(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
            num_pads=dec_input_num_pads)
        pad_mask = create_pad_mask(mask_size=torch.zeros(dec_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_row=dec_input_num_pads,
                                   pad_column=enc_input_num_pads)

        y = self.out_embedder(dec_input)
        pos_x = torch.arange(self.num_exp_dec).unsqueeze(0).expand(dec_input.size(0), self.num_exp_dec)
        pos_y = torch.arange(dec_input.size(1)).unsqueeze(0).expand(dec_input.size(0), dec_input.size(1))
        y = y + self.pos_encoder(pos_y)
        y_list = []
        for i, decoder in enumerate(self.decoders):
            y = decoder(x=y,
                        n_indexes=pos_x,
                        cross_connection_x=cross_input,
                        input_attention_mask=no_peak_and_pad_mask,
                        cross_attention_mask=pad_mask)
            y_list.append(y)
        y_list = torch.cat(y_list, dim=-1)
        y = y + self.out_dec_dropout(self.dec_reduce_group(y_list))
        y = self.dec_reduce_norm(y)

        y = self.vocab_linear(y)
        y = self.log_softmax(y)

        return y

    def forward(self, enc_x: torch.Tensor,
                enc_x_num_pads: torch.Tensor,
                sos_idx: int, eos_idx: int,
                max_seq_len: int):
        bs = enc_x.size(0)
        x = self.forward_enc(enc_input=enc_x, enc_input_num_pads=enc_x_num_pads)
        loop_pred = torch.ones((bs, 1)).type(torch.long) * sos_idx
        loop_logprobs = torch.zeros((bs, 1))
        for time_step in range(max_seq_len):
            log_probs = self.forward_dec(x, enc_x_num_pads, loop_pred[:, :time_step+1], torch.tensor([0]))
            topv, topi = log_probs[:, time_step, :].topk(k=1)
            loop_pred = torch.cat([loop_pred, topi], dim=-1)
            loop_logprobs = torch.cat([loop_logprobs, topv], dim=-1)
            if topi.item() == eos_idx:
                break
        return loop_pred, loop_logprobs


# Conversion function

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
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()

    img_size = 384
    with open(args.vocab_path, 'rb') as f:
        coco_tokens = pickle.load(f)

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

    # we test the generalization of the graph by testing on two images
    image_1 = preprocess_image(args.image_path_1)
    image_2 = preprocess_image(args.image_path_2)

    # Mode Args specification
    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model = End_ExpansionNet_v2_ONNX(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
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

    model.to('cpu')
    print("Loading model...")
    checkpoint = torch.load(args.load_model_path)
    partially_load_state_dict(model, checkpoint['model_state_dict'])

    print("Performing forwards...")
    model.eval()
    my_script = torch.jit.script(model)
    torch.onnx.export(
        my_script,
        (image_1, torch.tensor([0]),
         coco_tokens['word2idx_dict'][coco_tokens['sos_str']],
         coco_tokens['word2idx_dict'][coco_tokens['eos_str']],
         args.max_seq_len),
        args.output_onnx_path,
        input_names=['enc_x', 'enc_x_num_pads',
                     'sos_idx', 'eos_idx',
                     'max_seq_len'],
        output_names=['pred', 'logprobs'],
        export_params=True,  # questo serve per passare anche i parametri
        opset_version=14)
    print("ONNX graph conversion done. ONNX graph destination: " + args.output_onnx_path)

    onnx_model = onnx.load(args.output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX graph checked.")

    onnx_model = onnx.load(args.output_onnx_path)

    # generate optimized graph
    print("Testing firts image on ONNX runtime")
    ort_sess = ort.InferenceSession(args.output_onnx_path)
    input_dict = {'enc_x': image_1.numpy(),
                  'enc_x_num_pads': np.array([0]),
                  'sos_idx': np.array([coco_tokens['word2idx_dict'][coco_tokens['sos_str']]]),
                  'eos_idx': np.array([coco_tokens['word2idx_dict'][coco_tokens['eos_str']]]),
                  'max_seq_len': np.array([20])}
    outputs_ort = ort_sess.run(None, input_dict)
    output_caption = [coco_tokens['idx2word_list'][idx] for idx in outputs_ort[0][0]]
    print("\n\n\nONNX Runtime result:\n\t\t" + str(' '.join(output_caption)), end="\n\n\n")

    print("Testing second image on ONNX runtime")
    ort_sess = ort.InferenceSession(args.output_onnx_path)
    input_dict = {'enc_x': image_2.numpy(),
                  'enc_x_num_pads': np.array([0]),
                  'sos_idx': np.array([coco_tokens['word2idx_dict'][coco_tokens['sos_str']]]),
                  'eos_idx': np.array([coco_tokens['word2idx_dict'][coco_tokens['eos_str']]]),
                  'max_seq_len': np.array([20])}
    outputs_ort = ort_sess.run(None, input_dict)
    output_caption = [coco_tokens['idx2word_list'][idx] for idx in outputs_ort[0][0]]
    print("\n\n\nONNX Runtime result:\n\t\t" + str(' '.join(output_caption)), end="\n\n\n")

    print("Testing on ONNX-TensorRT backend")
    engine = backend.prepare(onnx_model, device='CUDA:0')
    input_data = (image_1, torch.tensor([0]),
                  torch.tensor([coco_tokens['word2idx_dict'][coco_tokens['sos_str']]]),
                  torch.tensor([coco_tokens['word2idx_dict'][coco_tokens['eos_str']]]),
                  torch.tensor([20]))
    output_data = engine.run(input_data)[0]
    print(output_data)
    print(output_data.shape)
    print("Closing.")
