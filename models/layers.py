import torch
import torch.nn as nn
import math

import numpy as np
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, dropout_perc):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_perc)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.dropout(self.embed(x)) * math.sqrt(float(self.d_model))


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

        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / np.sqrt(self.d_model)
        z = self.Z_dropout(z)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(mask == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mask == 0, 0.0)
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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_enc_exp_list, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
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

    def forward(self, x, n_indexes, mask):
        bs, dec_len, _ = x.shape

        cond = self.cond_embed(x).view(bs, dec_len, 1, self.d_model)
        query_exp = self.query_exp_vectors(n_indexes).unsqueeze(1)
        bias_exp = self.bias_exp_vectors(n_indexes).unsqueeze(1)
        query_exp = (query_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)
        bias_exp = (bias_exp + cond).view(bs, dec_len * self.num_exp, self.d_model)

        x_key = self.key_linear(x)
        z = torch.matmul(query_exp, x_key.transpose(-1, -2)) / np.sqrt(self.d_model)
        z = self.Z_dropout(z)

        mod_mask_1 = mask.unsqueeze(2).expand(bs, dec_len, self.num_exp, dec_len).contiguous(). \
            view(bs, dec_len * self.num_exp, dec_len)

        class_a_fw = F.relu(z)
        class_b_fw = F.relu(-z)
        class_a_fw = class_a_fw.masked_fill(mod_mask_1 == 0, 0.0)
        class_b_fw = class_b_fw.masked_fill(mod_mask_1 == 0, 0.0)
        class_a_fw = class_a_fw / (class_a_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_fw = class_b_fw / (class_b_fw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_fw, self.class_a_embed(x))
        class_b = torch.matmul(class_b_fw, self.class_b_embed(x))
        class_a = self.dropout_class_a_fw(class_a)
        class_b = self.dropout_class_b_fw(class_b)

        mod_mask_2 = mask.unsqueeze(-1).expand(bs, dec_len, dec_len, self.num_exp).contiguous(). \
            view(bs, dec_len, dec_len * self.num_exp)

        class_a_bw = F.relu(z.transpose(-2, -1))
        class_b_bw = F.relu(-z.transpose(-2, -1))
        class_a_bw = class_a_bw.masked_fill(mod_mask_2 == 0, 0.0)
        class_b_bw = class_b_bw.masked_fill(mod_mask_2 == 0, 0.0)
        class_a_bw = class_a_bw / (class_a_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_b_bw = class_b_bw / (class_b_bw.sum(dim=-1, keepdim=True) + self.eps)
        class_a = torch.matmul(class_a_bw, class_a + bias_exp)
        class_b = torch.matmul(class_b_bw, class_b + bias_exp)
        class_a = self.dropout_class_a_bw(class_a)
        class_b = self.dropout_class_b_bw(class_b)

        selector = torch.sigmoid(self.selector_embed(x))
        x_result = selector * class_a + (1 - selector) * class_b

        return x_result


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_exp, dropout_perc, eps=1e-9):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.mha = MultiHeadAttention(d_model, num_heads, dropout_perc)
        self.dyn_exp = DynamicExpansionBlock(d_model, num_exp, dropout_perc, eps)
        self.ff = FeedForward(d_model, d_ff, dropout_perc)

    def forward(self, x, n_indexes, cross_connection_x, input_attention_mask, cross_attention_mask):

        # Pre-LayerNorm
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.dyn_exp(x=x2, n_indexes=n_indexes, mask=input_attention_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.mha(q=x2, k=cross_connection_x, v=cross_connection_x,
                                        mask=cross_attention_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout_perc):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "num heads must be multiple of d_model"

        self.d_model = d_model
        self.d_k = int(d_model / num_heads)
        self.num_heads = num_heads

        self.Wq = nn.Linear(d_model, self.d_k * num_heads)
        self.Wk = nn.Linear(d_model, self.d_k * num_heads)
        self.Wv = nn.Linear(d_model, self.d_k * num_heads)

        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
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

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            sim_scores = sim_scores.masked_fill(mask == 0, value=-1e4)
        sim_scores = F.softmax(input=sim_scores, dim=-1)

        attention_applied = torch.matmul(sim_scores, v_proj)
        attention_applied_concatenated = attention_applied.permute(0, 2, 1, 3).contiguous()\
            .view(batch_size, q_seq_len, self.d_model)

        out = self.out_linear(attention_applied_concatenated)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_perc)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
