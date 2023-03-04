import torch
from models.layers import EmbeddingLayer, DecoderLayer, EncoderLayer
from utils.masking import create_pad_mask, create_no_peak_and_pad_mask
from models.captioning_model import CaptioningModel
from models.swin_transformer_mod import SwinTransformer

import torch.nn as nn


class End_ExpansionNet_v2(CaptioningModel):
    def __init__(self,

                 # swin transf
                 swin_img_size, swin_patch_size, swin_in_chans,
                 swin_embed_dim, swin_depths, swin_num_heads,
                 swin_window_size, swin_mlp_ratio, swin_qkv_bias, swin_qk_scale,
                 swin_drop_rate, swin_attn_drop_rate, swin_drop_path_rate,
                 swin_norm_layer, swin_ape, swin_patch_norm,
                 swin_use_checkpoint,

                 # linear_size,
                 final_swin_dim,

                 # captioning
                 d_model, N_enc, N_dec, ff, num_heads, num_exp_enc_list, num_exp_dec,
                 output_word2idx, output_idx2word, max_seq_len, drop_args, rank=0):
        super(End_ExpansionNet_v2, self).__init__()

        self.swin_transf = SwinTransformer(
                 img_size=swin_img_size, patch_size=swin_patch_size, in_chans=swin_in_chans,
                 embed_dim=swin_embed_dim, depths=swin_depths, num_heads=swin_num_heads,
                 window_size=swin_window_size, mlp_ratio=swin_mlp_ratio, qkv_bias=swin_qkv_bias, qk_scale=swin_qk_scale,
                 drop_rate=swin_drop_rate, attn_drop_rate=swin_attn_drop_rate, drop_path_rate=swin_drop_path_rate,
                 norm_layer=swin_norm_layer, ape=swin_ape, patch_norm=swin_patch_norm,
                 use_checkpoint=swin_use_checkpoint)

        self.output_word2idx = output_word2idx
        self.output_idx2word = output_idx2word
        self.max_seq_len = max_seq_len

        self.num_exp_dec = num_exp_dec
        self.num_exp_enc_list = num_exp_enc_list

        self.N_enc = N_enc
        self.N_dec = N_dec
        self.d_model = d_model

        self.encoders = nn.ModuleList([EncoderLayer(d_model, ff, num_exp_enc_list, drop_args.enc) for _ in range(N_enc)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, num_heads, ff, num_exp_dec, drop_args.dec) for _ in range(N_dec)])

        self.input_embedder_dropout = nn.Dropout(drop_args.enc_input)
        self.input_linear = torch.nn.Linear(final_swin_dim, d_model)
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

        self.check_required_attributes()

    def forward_enc(self, enc_input, enc_input_num_pads):

        assert (enc_input_num_pads is None or enc_input_num_pads == ([0] * enc_input.size(0))), \
            "End to End case have no padding"
        x = self.swin_transf(enc_input)

        enc_input = self.input_embedder_dropout(self.input_linear(x))
        x = enc_input
        enc_input_num_pads = [0] * enc_input.size(0)

        max_num_enc = sum(self.num_exp_enc_list)
        pos_x = torch.arange(max_num_enc).unsqueeze(0).expand(enc_input.size(0), max_num_enc).to(self.rank)
        pad_mask = create_pad_mask(mask_size=(enc_input.size(0), max_num_enc, enc_input.size(1)),
                                   pad_row=[0] * enc_input.size(0),
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)

        x_list = []
        for i in range(self.N_enc):
            x = self.encoders[i](x=x, n_indexes=pos_x, mask=pad_mask)
            x_list.append(x)
        x_list = torch.cat(x_list, dim=-1)
        x = x + self.out_enc_dropout(self.enc_reduce_group(x_list))
        x = self.enc_reduce_norm(x)

        return x

    def forward_dec(self, cross_input, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=False):
        assert (enc_input_num_pads is None or enc_input_num_pads == ([0] * cross_input.size(0))), \
            "enc_input_num_pads should be no None"

        enc_input_num_pads = [0] * dec_input.size(0)
        no_peak_and_pad_mask = create_no_peak_and_pad_mask(
                                mask_size=(dec_input.size(0), dec_input.size(1), dec_input.size(1)),
                                num_pads=dec_input_num_pads,
                                rank=self.rank)
        pad_mask = create_pad_mask(mask_size=(dec_input.size(0), dec_input.size(1), cross_input.size(1)),
                                   pad_row=dec_input_num_pads,
                                   pad_column=enc_input_num_pads,
                                   rank=self.rank)

        y = self.out_embedder(dec_input)
        pos_x = torch.arange(self.num_exp_dec).unsqueeze(0).expand(dec_input.size(0), self.num_exp_dec).to(self.rank)
        pos_y = torch.arange(dec_input.size(1)).unsqueeze(0).expand(dec_input.size(0), dec_input.size(1)).to(self.rank)
        y = y + self.pos_encoder(pos_y)
        y_list = []
        for i in range(self.N_dec):
            y = self.decoders[i](x=y,
                                 n_indexes=pos_x,
                                 cross_connection_x=cross_input,
                                 input_attention_mask=no_peak_and_pad_mask,
                                 cross_attention_mask=pad_mask)
            y_list.append(y)
        y_list = torch.cat(y_list, dim=-1)
        y = y + self.out_dec_dropout(self.dec_reduce_group(y_list))
        y = self.dec_reduce_norm(y)

        y = self.vocab_linear(y)

        if apply_log_softmax:
            y = self.log_softmax(y)

        return y


    def get_batch_multiple_sampled_prediction(self, enc_input, enc_input_num_pads, num_outputs,
                                              sos_idx, eos_idx, max_seq_len):

        bs = enc_input.size(0)
        x = self.forward_enc(enc_input=enc_input, enc_input_num_pads=enc_input_num_pads)
        enc_seq_len = x.size(1)
        x = x.unsqueeze(1).expand(-1, num_outputs, -1, -1).reshape(bs * num_outputs, enc_seq_len, x.shape[-1])

        upperbound_vector = torch.tensor([max_seq_len] * bs * num_outputs, dtype=torch.int).to(self.rank)
        where_is_eos_vector = upperbound_vector.clone()
        eos_vector = torch.tensor([eos_idx] * bs * num_outputs, dtype=torch.long).to(self.rank)
        finished_flag_vector = torch.zeros(bs * num_outputs).type(torch.int)

        predicted_caption = torch.tensor([sos_idx] * (bs * num_outputs), dtype=torch.long).to(self.rank).unsqueeze(-1)
        predicted_caption_prob = torch.zeros(bs * num_outputs).to(self.rank).unsqueeze(-1)

        dec_input_num_pads = [0]*(bs*num_outputs)
        time_step = 0
        while (finished_flag_vector.sum() != bs * num_outputs) and time_step < max_seq_len:
            dec_input = predicted_caption
            log_probs = self.forward_dec(x, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=True)

            prob_dist = torch.distributions.Categorical(torch.exp(log_probs[:, time_step]))
            sampled_word_indexes = prob_dist.sample()

            predicted_caption = torch.cat((predicted_caption, sampled_word_indexes.unsqueeze(-1)), dim=-1)
            predicted_caption_prob = torch.cat((predicted_caption_prob,
                log_probs[:, time_step].gather(index=sampled_word_indexes.unsqueeze(-1), dim=-1)), dim=-1)
            time_step += 1

            where_is_eos_vector = torch.min(where_is_eos_vector,
                                    upperbound_vector.masked_fill(sampled_word_indexes == eos_vector, time_step))
            finished_flag_vector = torch.max(finished_flag_vector,
                                             (sampled_word_indexes == eos_vector).type(torch.IntTensor))

        res_predicted_caption = []
        for i in range(bs):
            res_predicted_caption.append([])
            for j in range(num_outputs):
                index = i*num_outputs + j
                res_predicted_caption[i].append(
                    predicted_caption[index, :where_is_eos_vector[index].item()+1].tolist())

        where_is_eos_vector = where_is_eos_vector.unsqueeze(-1).expand(-1, time_step+1)
        arange_tensor = torch.arange(time_step+1).unsqueeze(0).expand(bs * num_outputs, -1).to(self.rank)
        predicted_caption_prob.masked_fill_(arange_tensor > where_is_eos_vector, 0.0)
        res_predicted_caption_prob = predicted_caption_prob.reshape(bs, num_outputs, -1)

        return res_predicted_caption, res_predicted_caption_prob
