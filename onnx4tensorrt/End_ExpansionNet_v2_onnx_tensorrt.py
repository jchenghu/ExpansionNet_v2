
import torch
import torch.nn as nn

from models.layers import EmbeddingLayer, EncoderLayer, DecoderLayer
from onnx4tensorrt.swin_transformer_onnx_tensorrt import SwinTransformer_ONNX_TensorRT


def create_pad_mask(mask_size, pad_row, pad_column):
    bs, out_len, in_len = mask_size.shape
    pad_row_tens = torch.tensor([out_len]) - pad_row.unsqueeze(-1).repeat(1, out_len)
    pad_col_tens = torch.tensor([in_len]) - pad_column.unsqueeze(-1).repeat(1, in_len)
    # cast to int to extract boolean values
    arange_on_columns = (torch.arange(in_len).unsqueeze(0).repeat(bs, 1) < pad_col_tens).type(torch.int32)
    arange_on_rows = (torch.arange(out_len).unsqueeze(0).repeat(bs, 1) < pad_row_tens).type(torch.int32)
    # cast to float for tensorrtx
    arange_on_columns = arange_on_columns.type(torch.float32)
    arange_on_rows = arange_on_rows.type(torch.float32)
    mask = torch.matmul(arange_on_rows.unsqueeze(-1), arange_on_columns.unsqueeze(-2))
    return mask


def create_no_peak_and_pad_mask(mask_size, num_pads):
    block_mask = create_pad_mask(mask_size, num_pads, num_pads)
    bs, seq_len, seq_len = mask_size.shape
    column_const = torch.arange(seq_len).unsqueeze(-1).repeat(1, seq_len)
    row_const = torch.arange(seq_len).unsqueeze(0).repeat(seq_len, 1)
    triang_mask = (column_const >= row_const).type(torch.int32)
    triang_mask = triang_mask.unsqueeze(0).repeat(bs, 1, 1).type(torch.float)
    # tril is not currently supported by tensorrt
    # triang_mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.float),
    #                          diagonal=0).unsqueeze(0).repeat(bs, 1, 1)
    return torch.mul(block_mask, triang_mask)


class End_ExpansionNet_v2_ONNX_TensorRT(nn.Module):

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
        super(End_ExpansionNet_v2_ONNX_TensorRT, self).__init__()

        # swin
        self.swin_transf = SwinTransformer_ONNX_TensorRT(
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

    def single_step(self, time_step: int, x: torch.Tensor,
                    enc_x_num_pads: torch.Tensor, loop_pred: torch.Tensor, loop_logprobs: torch.Tensor):
        log_probs = self.forward_dec(x, enc_x_num_pads, loop_pred[:, :time_step + 1], torch.tensor([0]))
        topv, topi = log_probs[:, time_step, :].topk(k=1)
        loop_pred = torch.cat([loop_pred, topi + enc_x_num_pads], dim=-1)
        loop_logprobs = torch.cat([loop_logprobs, topv], dim=-1)
        return loop_pred, loop_logprobs

    def forward(self, enc_x: torch.Tensor,
                enc_x_num_pads: torch.Tensor,
                sos_idx: torch.Tensor):
        bs = enc_x.size(0)
        x = self.forward_enc(enc_input=enc_x, enc_input_num_pads=enc_x_num_pads)
        loop_pred = torch.ones((bs, 1)).type(torch.int32) * sos_idx
        loop_logprobs = torch.zeros((bs, 1))

        # TensorRT friendly implementation of a loop
        # TO-DO: check whether is detrimental to performances such implementation
        loop_pred, loop_logprobs = self.single_step(0, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(1, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(2, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(3, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(4, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(5, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(6, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(7, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(8, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(9, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(10, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(11, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(12, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(13, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(14, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(15, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(16, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(17, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(18, x, enc_x_num_pads, loop_pred, loop_logprobs)
        loop_pred, loop_logprobs = self.single_step(19, x, enc_x_num_pads, loop_pred, loop_logprobs)

        return loop_pred, loop_logprobs

