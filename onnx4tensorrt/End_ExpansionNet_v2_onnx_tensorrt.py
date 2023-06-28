
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import EmbeddingLayer, FeedForward

from onnx4tensorrt.swin_transformer_onnx_tensorrt import SwinTransformer_ONNX_TensorRT


NUM_FEATURES = 144
MAX_DECODE_STEPS = 20


def create_pad_mask(mask_size, pad_row, pad_column):
    bs, out_len, in_len = mask_size
    pad_row_tens = torch.tensor([out_len]) - torch.tensor(pad_row).unsqueeze(-1).repeat(1, out_len)
    pad_col_tens = torch.tensor([in_len]) - torch.tensor(pad_column).unsqueeze(-1).repeat(1, in_len)
    arange_on_columns = (torch.arange(in_len).unsqueeze(0).repeat(bs, 1) < pad_col_tens).type(torch.int32)
    arange_on_rows = (torch.arange(out_len).unsqueeze(0).repeat(bs, 1) < pad_row_tens).type(torch.int32)
    mask = torch.matmul(arange_on_rows.unsqueeze(-1), arange_on_columns.unsqueeze(-2))
    return mask


def create_no_peak_and_pad_mask(mask_size, num_pads):
    block_mask = create_pad_mask(mask_size, num_pads, num_pads)
    bs, seq_len, seq_len = mask_size
    triang_mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.float),
                             diagonal=0).unsqueeze(0).repeat(bs, 1, 1)
    return torch.mul(block_mask, triang_mask)


class DynamicExpansionBlock(nn.Module):
    def __init__(self, d_model, num_exp, dropout_perc, eps):
        super().__init__()
        self.d_model = d_model

        self.num_exp = num_exp
        self.cond_embed = nn.Linear(d_model, d_model)

        self.query_exp_vectors = nn.Embedding(self.num_exp, d_model)
        self.bias_exp_vectors = nn.Embedding(self.num_exp, d_model)

        self.key_linear = nn.Linear(d_model, d_model)
        self.class_a_embed = nn.Linear(d_model, d_model)
        self.class_b_embed = nn.Linear(d_model, d_model)

        self.selector_embed = nn.Linear(d_model, d_model)

        self.dropout_class_a_fw = nn.Dropout(dropout_perc)
        self.dropout_class_b_fw = nn.Dropout(dropout_perc)
        self.dropout_class_a_bw = nn.Dropout(dropout_perc)
        self.dropout_class_b_bw = nn.Dropout(dropout_perc)

        self.Z_dropout = nn.Dropout(dropout_perc)

        self.eps = eps

    def forward(self, x, n_indexes, fw_mask, bw_mask):
        bs, dec_len, _ = x.shape

        cond = self.cond_embed(x).view(bs, dec_len, 1, self.d_model)
        query_exp = self.query_exp_vectors(n_indexes).unsqueeze(1)
        bias_exp = self.bias_exp_vectors(n_indexes).unsqueeze(1)
        query_exp = (query_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)
        bias_exp = (bias_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)

        x_key = self.key_linear(x)
        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / (self.d_model ** 0.5)
        z = self.Z_dropout(z)

        #mod_mask_1 = mask.unsqueeze(2).expand(bs, dec_len, self.num_exp, dec_len).contiguous(). \
        #    view(bs, dec_len * self.num_exp, dec_len)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(fw_mask == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(fw_mask == 0, 0.0)
        #class_a_fw = class_a_fw * fw_mask
        #class_b_fw = class_b_fw * fw_mask
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_fw, self.class_a_embed(x))
        class_b = torch.matmul(class_b_fw, self.class_b_embed(x))
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        # mod_mask_2 = mask.unsqueeze(-1).expand(bs, dec_len, dec_len, self.num_exp).contiguous(). \
        #             view(bs, dec_len, dec_len * self.num_exp)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))
        class_a_bw = class_a_bw.masked_fill(bw_mask == 0, 0.0)
        class_b_bw = class_b_bw.masked_fill(bw_mask == 0, 0.0)
        #class_a_bw = class_a_bw * bw_mask
        #class_b_bw = class_b_bw * bw_mask
        class_a_bw = class_a_bw / (class_a_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_bw = class_b_bw / (class_b_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_bw, class_a + bias_exp)
        class_b = torch.matmul(class_b_bw, class_b + bias_exp)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b

        return x_result


class StaticExpansionBlock(nn.Module):
    def __init__(self, d_model, num_enc_exp_list, dropout_perc, eps):
        super().__init__()
        self.d_model = d_model
        self.num_enc_exp_list = num_enc_exp_list

        self.query_exp_vectors = nn.Embedding(sum(num_enc_exp_list), d_model)
        self.bias_exp_vectors = nn.Embedding(sum(num_enc_exp_list), d_model)

        self.key_embed = nn.Linear(d_model, d_model)
        self.class_a_embed = nn.Linear(d_model, d_model)
        self.class_b_embed = nn.Linear(d_model, d_model)

        self.selector_embed = nn.Linear(d_model, d_model)

        self.dropout_class_a_fw = nn.Dropout(dropout_perc)
        self.dropout_class_b_fw = nn.Dropout(dropout_perc)

        self.dropout_class_a_bw = nn.Dropout(dropout_perc)
        self.dropout_class_b_bw = nn.Dropout(dropout_perc)

        self.Z_dropout = nn.Dropout(dropout_perc)

        self.eps = eps

    def forward(self, x, n_indexes, mask):
        bs, enc_len, _ = x.shape

        query_exp = self.query_exp_vectors(n_indexes)
        bias_exp = self.bias_exp_vectors(n_indexes)
        x_key = self.key_embed(x)

        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / (self.d_model ** 0.5)
        z = self.Z_dropout(z)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        # this operation is disliked by TensorRT FP16 for some reason.
        class_a_fw = class_a_fw.masked_fill(mask == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mask == 0, 0.0)
        #class_a_fw = class_a_fw * mask
        #class_b_fw = class_b_fw * mask
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)

        class_a = torch.matmul(class_a_fw, self.class_a_embed(x)) + bias_exp
        class_b = torch.matmul(class_b_fw, self.class_b_embed(x)) + bias_exp
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))

        accum = 0
        class_a_bw_list = []
        class_b_bw_list = []
        for j in range(len(self.num_enc_exp_list)):
            from_idx = accum
            to_idx = accum + self.num_enc_exp_list[j]
            accum += self.num_enc_exp_list[j]
            class_a_bw_list.append(class_a_bw[:, :, from_idx:to_idx] / (class_a_bw[:, :, from_idx:to_idx].sum(dim=-1, keepdim=True) + self.eps))
            class_b_bw_list.append(class_b_bw[:, :, from_idx:to_idx] / (class_b_bw[:, :, from_idx:to_idx].sum(dim=-1, keepdim=True) + self.eps))
        class_a_bw = torch.cat(class_a_bw_list, dim=-1)
        class_b_bw = torch.cat(class_b_bw_list, dim=-1)

        class_a = torch.matmul(class_a_bw, class_a) / len(self.num_enc_exp_list)
        class_b = torch.matmul(class_b_bw, class_b) / len(self.num_enc_exp_list)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b

        return x_result


class EncoderLayer_ONNX(nn.Module):
    def __init__(self, d_model, d_ff, num_enc_exp_list, dropout_perc, eps=1e-4):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps)
        self.norm_2 = nn.LayerNorm(d_model, eps)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)

        self.stc_exp = StaticExpansionBlock(d_model, num_enc_exp_list, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.stc_exp(x=x2, n_indexes=n_indexes, mask=mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class MultiHeadAttention_ONNX(nn.Module):
    def __init__(self, d_model, num_heads, dropout_perc):
        super(MultiHeadAttention_ONNX, self).__init__()
        assert d_model % num_heads == 0, "num heads must be multiple of d_model"

        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, self.d_k * num_heads)
        self.Wk = nn.Linear(d_model, self.d_k * num_heads)
        self.Wv = nn.Linear(d_model, self.d_k * num_heads)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        batch_size, q_seq_len, _ = q.shape
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)

        k_proj = self.Wk(k).view(batch_size, k_seq_len, self.num_heads, self.d_k)
        q_proj = self.Wq(q).view(batch_size, q_seq_len, self.num_heads, self.d_k)
        v_proj = self.Wv(v).view(batch_size, v_seq_len, self.num_heads, self.d_k)

        k_proj = k_proj.transpose(2, 1)
        q_proj = q_proj.transpose(2, 1)
        v_proj = v_proj.transpose(2, 1)

        sim_scores = torch.matmul(q_proj, k_proj.transpose(3, 2))
        sim_scores = sim_scores / self.d_k ** 0.5

        # mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) -> it is repeated outside
        #sim_scores = sim_scores.masked_fill(mask == 0, value=-1e4)
        sim_scores = sim_scores.masked_fill(mask == 1, value=-1e3)
        #sim_scores = sim_scores + mask * (-1e4)
        sim_scores = F.softmax(input=sim_scores, dim=-1)
        attention_applied = torch.matmul(sim_scores, v_proj)
        attention_applied_concatenated = attention_applied.permute(0, 2, 1, 3).contiguous()\
            .view(batch_size, q_seq_len, self.d_model)

        out = self.out_linear(attention_applied_concatenated)
        return out


class DecoderLayer_ONNX(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_exp, dropout_perc, eps=1e-4):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps)
        self.norm_2 = nn.LayerNorm(d_model, eps)
        self.norm_3 = nn.LayerNorm(d_model, eps)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha = MultiHeadAttention_ONNX(d_model, num_heads, dropout_perc)
        self.dyn_exp = DynamicExpansionBlock(d_model, num_exp, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, cross_connection_x, fw_mask, bw_mask, cross_attention_mask):

        # Pre-LayerNorm
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.dyn_exp(x=x2, n_indexes=n_indexes,
                                            fw_mask=fw_mask,
                                            bw_mask=bw_mask))

        x2 = self.norm_2(x)
        #x = x2 + (cross_connection_x * 0.0001).sum(dim=1, keepdim=True) + \
        #    cross_attention_mask.sum() / cross_attention_mask.sum()
        x = x + self.dropout_2(self.mha(q=x2, k=cross_connection_x, v=cross_connection_x,
                                        mask=cross_attention_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))

        return x


class End_ExpansionNet_v2_ONNX_TensorRT(nn.Module):
    MAX_DECODE_STEPS = 20
    NUM_FEATURES = 144

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
            [EncoderLayer_ONNX(d_model, ff, num_exp_enc_list, drop_args.enc) for _ in range(N_enc)])
        self.decoders = nn.ModuleList(
            [DecoderLayer_ONNX(d_model, num_heads, ff, num_exp_dec, drop_args.dec) for _ in range(N_dec)])

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

    def forward_enc(self, enc_input: torch.Tensor, enc_mask: torch.Tensor):

        x = self.swin_transf(enc_input)
        enc_input = self.input_embedder_dropout(self.input_linear(x))
        x = enc_input

        max_num_enc = sum(self.num_exp_enc_list)
        pos_x = torch.arange(max_num_enc).unsqueeze(0).expand(enc_input.size(0), max_num_enc)

        x_list = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x=x, n_indexes=pos_x, mask=enc_mask)
            x_list.append(x)
        x_list = torch.cat(x_list, dim=-1)
        x = x + self.out_enc_dropout(self.enc_reduce_group(x_list))
        x = self.enc_reduce_norm(x)
        return x

    def forward_dec(self, cross_input: torch.Tensor, dec_input: torch.Tensor,
                    fw_dec_mask: torch.Tensor,
                    bw_dec_mask: torch.Tensor,
                    cross_mask: torch.Tensor):

        y = self.out_embedder(dec_input)
        pos_x = torch.arange(self.num_exp_dec).unsqueeze(0).expand(dec_input.size(0), self.num_exp_dec)
        pos_y = torch.arange(dec_input.size(1)).unsqueeze(0).expand(dec_input.size(0), dec_input.size(1))
        y = y + self.pos_encoder(pos_y)

        y_list = []
        for i, decoder in enumerate(self.decoders):
            y = decoder(x=y, n_indexes=pos_x,
                        cross_connection_x=cross_input,
                        fw_mask=fw_dec_mask,
                        bw_mask=bw_dec_mask,
                        cross_attention_mask=cross_mask)
            y_list.append(y)
        y_list = torch.cat(y_list, dim=-1)
        y = y + self.out_dec_dropout(self.dec_reduce_group(y_list))
        y = self.dec_reduce_norm(y)

        y = self.vocab_linear(y)

        y = self.log_softmax(y)
        return y

    def single_step(self, time_step: int, x: torch.Tensor,
                    loop_pred: torch.Tensor, loop_logprobs: torch.Tensor,
                    fw_dec_mask: torch.Tensor,
                    bw_dec_mask: torch.Tensor,
                    cross_mask: torch.Tensor
                    ):
        log_probs = self.forward_dec(x, loop_pred[:, :time_step + 1],
                                     fw_dec_mask, bw_dec_mask, cross_mask)
        # log_probs *= 0.001  # scale by a factor M, to mitigate FP16 underrepresentation
        topv, topi = log_probs[:, time_step, :].topk(k=1)

        loop_pred = torch.cat([loop_pred, topi], dim=-1)
        loop_logprobs = torch.cat([loop_logprobs, topv], dim=-1)
        return loop_pred, loop_logprobs

    def forward(self, enc_x: torch.Tensor, sos_idx: torch.Tensor,
                enc_mask: torch.Tensor,
                fw_dec_mask: torch.Tensor,
                bw_dec_mask: torch.Tensor,
                cross_mask: torch.Tensor):

        bs = enc_x.size(0)
        enc_x = self.forward_enc(enc_input=enc_x, enc_mask=enc_mask)
        loop_pred = torch.ones((bs, 1)).type(torch.int32) * sos_idx
        loop_logprobs = torch.zeros((bs, 1))

        # TensorRT friendly implementation of a loop
        # TO-DO: check whether is detrimental to performances such implementation
        step = 1 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(0, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:1],
                                                    bw_dec_mask[:, 0:1, 0:step],
                                                    cross_mask[:, :, 0:1, :])
        step = 2 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(1, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:2],
                                                    bw_dec_mask[:, 0:2, 0:step],
                                                    cross_mask[:, :, 0:2, :])
        step = 3 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(2, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:3],
                                                    bw_dec_mask[:, 0:3, 0:step],
                                                    cross_mask[:, :, 0:3, :])

        step = 4 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(3, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:4],
                                                    bw_dec_mask[:, 0:4, 0:step],
                                                    cross_mask[:, :, 0:4, :])
        step = 5 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(4, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:5],
                                                    bw_dec_mask[:, 0:5, 0:step],
                                                    cross_mask[:, :, 0:5, :])
        step = 6 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(5, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:6],
                                                    bw_dec_mask[:, 0:6, 0:step],
                                                    cross_mask[:, :, 0:6, :])
        step = 7 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(6, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:7],
                                                    bw_dec_mask[:, 0:7, 0:step],
                                                    cross_mask[:, :, 0:7, :])
        step = 8 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(7, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:8],
                                                    bw_dec_mask[:, 0:8, 0:step],
                                                    cross_mask[:, :, 0:8, :])
        step = 9 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(8, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:9],
                                                    bw_dec_mask[:, 0:9, 0:step],
                                                    cross_mask[:, :, 0:9, :])
        step = 10 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(9, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:10],
                                                    bw_dec_mask[:, 0:10, 0:step],
                                                    cross_mask[:, :, 0:10, :])
        step = 11 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(10, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:11],
                                                    bw_dec_mask[:, 0:11, 0:step],
                                                    cross_mask[:, :, 0:11, :])
        step = 12 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(11, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:12],
                                                    bw_dec_mask[:, 0:12, 0:step],
                                                    cross_mask[:, :, 0:12, :])
        step = 13 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(12, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:13],
                                                    bw_dec_mask[:, 0:13, 0:step],
                                                    cross_mask[:, :, 0:13, :])
        step = 14 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(13, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:14],
                                                    bw_dec_mask[:, 0:14, 0:step],
                                                    cross_mask[:, :, 0:14, :])
        step = 15 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(14, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:15],
                                                    bw_dec_mask[:, 0:15, 0:step],
                                                    cross_mask[:, :, 0:15, :])
        step = 16 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(15, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:16],
                                                    bw_dec_mask[:, 0:16, 0:step],
                                                    cross_mask[:, :, 0:16, :])
        step = 17 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(16, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:17],
                                                    bw_dec_mask[:, 0:17, 0:step],
                                                    cross_mask[:, :, 0:17, :])
        step = 18 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(17, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:18],
                                                    bw_dec_mask[:, 0:18, 0:step],
                                                    cross_mask[:, :, 0:18, :])
        step = 19 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(18, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:19],
                                                    bw_dec_mask[:, 0:19, 0:step],
                                                    cross_mask[:, :, 0:19, :])
        step = 20 * self.num_exp_dec
        loop_pred, loop_logprobs = self.single_step(19, enc_x, loop_pred, loop_logprobs,
                                                    fw_dec_mask[:, 0:step, 0:20],
                                                    bw_dec_mask[:, 0:20, 0:step],
                                                    cross_mask[:, :, 0:20, :])
        return loop_pred, loop_logprobs

