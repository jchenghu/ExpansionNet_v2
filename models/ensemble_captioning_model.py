
import torch
import torch.nn as nn
from models.captioning_model import CaptioningModel


class EsembleCaptioningModel(CaptioningModel):
    def __init__(self, models_list, rank):
        super().__init__()
        self.num_models = len(models_list)
        self.models_list = models_list
        self.rank = rank

        self.dummy_linear = nn.Linear(1, 1)

        for model in self.models_list:
            model.eval()

    def forward(self, enc_x, dec_x=None,
                enc_x_num_pads=[0], dec_x_num_pads=[0], apply_log_softmax=False,
                mode='beam_search', **kwargs):
        assert (mode == 'beam_search'), "this class supports only beam search."
        sos_idx = kwargs.get('sos_idx', -999)
        eos_idx = kwargs.get('eos_idx', -999)
        if mode == 'beam_search':
            beam_size_arg = kwargs.get('beam_size', 5)
            how_many_outputs_per_beam = kwargs.get('how_many_outputs', 1)
            beam_max_seq_len = kwargs.get('beam_max_seq_len', 20)
            sample_or_max = kwargs.get('sample_or_max', 'max')
            out_classes, out_logprobs = self.ensemble_beam_search(
                enc_x, enc_x_num_pads,
                beam_size=beam_size_arg,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                how_many_outputs=how_many_outputs_per_beam,
                max_seq_len=beam_max_seq_len,
                sample_or_max=sample_or_max)
            return out_classes, out_logprobs

    def forward_enc(self, enc_input, enc_input_num_pads):
        x_outputs_list = []
        for i in range(self.num_models):
            x_outputs = self.models_list[i].forward_enc(enc_input, enc_input_num_pads)
            x_outputs_list.append(x_outputs)
        return x_outputs_list

    def forward_dec(self, cross_input_list, enc_input_num_pads, dec_input, dec_input_num_pads, apply_log_softmax=False):

        import torch.nn.functional as F
        y_outputs = []
        for i in range(self.num_models):
            y_outputs.append(
                F.softmax(self.models_list[i].forward_dec(
                    cross_input_list[i], enc_input_num_pads,
                    dec_input, dec_input_num_pads, False).unsqueeze(0), dim=-1))
        avg = torch.cat(y_outputs, dim=0).mean(dim=0).log()

        return avg

    # quite unclean coding, to be re-factored in the future...
    # since it's a bit similar to the single model case
    def ensemble_beam_search(self, enc_input, enc_input_num_pads, sos_idx, eos_idx,
                             beam_size=3, how_many_outputs=1, max_seq_len=20, sample_or_max='max',):
        assert (how_many_outputs <= beam_size), "requested output per sequence must be lower than beam width"
        assert (sample_or_max == 'max' or sample_or_max == 'sample'), "argument must be chosen between \'max\' and \'sample\'"
        bs = enc_input.shape[0]

        # the cross_dec_input is computed once
        cross_enc_output_list = self.forward_enc(enc_input, enc_input_num_pads)

        # init: ------------------------------------------------------------------
        init_dec_class = torch.tensor([sos_idx] * bs).unsqueeze(1).type(torch.long).to(self.rank)
        init_dec_logprob = torch.tensor([0.0] * bs).unsqueeze(1).type(torch.float).to(self.rank)
        log_probs = self.forward_dec(cross_input_list=cross_enc_output_list, enc_input_num_pads=enc_input_num_pads,
                                     dec_input=init_dec_class, dec_input_num_pads=[0] * bs,
                                     apply_log_softmax=True)
        if sample_or_max == 'max':
            _, topi = torch.topk(log_probs, k=beam_size, sorted=True)
        else:  # sample
            topi = torch.exp(log_probs[:, 0, :]).multinomial(num_samples=beam_size, replacement=False)
            topi = topi.unsqueeze(1)

        init_dec_class = init_dec_class.repeat(1, beam_size)
        init_dec_class = init_dec_class.unsqueeze(-1)
        top_beam_size_class = topi.transpose(-2, -1)
        init_dec_class = torch.cat((init_dec_class, top_beam_size_class), dim=-1)

        init_dec_logprob = init_dec_logprob.repeat(1, beam_size)
        init_dec_logprob = init_dec_logprob.unsqueeze(-1)
        top_beam_size_logprob = log_probs.gather(dim=-1, index=topi)
        top_beam_size_logprob = top_beam_size_logprob.transpose(-2, -1)
        init_dec_logprob = torch.cat((init_dec_logprob, top_beam_size_logprob), dim=-1)

        tmp_cross_enc_output_list = []
        for cross_enc_output in cross_enc_output_list:
            bs, enc_seq_len, d_model = cross_enc_output.shape
            cross_enc_output = cross_enc_output.unsqueeze(1)
            cross_enc_output = cross_enc_output.expand(-1, beam_size, -1, -1)
            cross_enc_output = cross_enc_output.reshape(bs * beam_size, enc_seq_len, d_model).contiguous()
            tmp_cross_enc_output_list.append(cross_enc_output)
        cross_enc_output_list = tmp_cross_enc_output_list
        enc_input_num_pads = [enc_input_num_pads[i] for i in range(bs) for _ in range(beam_size)]

        loop_dec_classes = init_dec_class
        loop_dec_logprobs = init_dec_logprob
        loop_cumul_logprobs = loop_dec_logprobs.sum(dim=-1, keepdims=True)

        loop_num_elem_vector = torch.tensor([2] * (bs * beam_size)).to(self.rank)

        for time_step in range(2, max_seq_len):
            loop_dec_classes = loop_dec_classes.reshape(bs * beam_size, time_step).contiguous()

            log_probs = self.forward_dec(cross_input_list=cross_enc_output_list, enc_input_num_pads=enc_input_num_pads,
                                         dec_input=loop_dec_classes,
                                         dec_input_num_pads=(time_step-loop_num_elem_vector).tolist(),
                                         apply_log_softmax=True)
            if sample_or_max == 'max':
                _, topi = torch.topk(log_probs[:, time_step-1, :], k=beam_size, sorted=True)
            else:  # sample
                topi = torch.exp(log_probs[:, time_step-1, :]).multinomial(num_samples=beam_size,
                                                                           replacement=False)

            top_beam_size_word_classes = topi.reshape(bs, beam_size, beam_size)

            top_beam_size_word_logprobs = log_probs[:, time_step-1, :].gather(dim=-1, index=topi)
            top_beam_size_word_logprobs = top_beam_size_word_logprobs.reshape(bs, beam_size, beam_size)

            there_is_eos_mask = (loop_dec_classes.view(bs, beam_size, time_step) == eos_idx). \
                sum(dim=-1, keepdims=True).type(torch.bool)

            top_beam_size_word_logprobs[:, :, 0:1].masked_fill_(there_is_eos_mask, 0.0)
            top_beam_size_word_logprobs[:, :, 1:].masked_fill_(there_is_eos_mask, -999.0)

            comparison_logprobs = loop_cumul_logprobs + top_beam_size_word_logprobs

            comparison_logprobs = comparison_logprobs.contiguous().view(bs, beam_size * beam_size)
            _, topi = torch.topk(comparison_logprobs, k=beam_size, sorted=True)
            which_sequence = topi // beam_size
            which_word = topi % beam_size

            loop_dec_classes = loop_dec_classes.view(bs, beam_size, -1)
            loop_dec_logprobs = loop_dec_logprobs.view(bs, beam_size, -1)

            bs_idxes = torch.arange(bs).unsqueeze(-1)
            new_loop_dec_classes = loop_dec_classes[[bs_idxes, which_sequence]]
            new_loop_dec_logprobs = loop_dec_logprobs[[bs_idxes, which_sequence]]

            which_sequence_top_beam_size_word_classes = top_beam_size_word_classes[[bs_idxes, which_sequence]]
            which_sequence_top_beam_size_word_logprobs = top_beam_size_word_logprobs[
                [bs_idxes, which_sequence]]
            which_word = which_word.unsqueeze(-1)

            lastword_top_beam_size_classes = which_sequence_top_beam_size_word_classes.gather(dim=-1,
                                                                                              index=which_word)
            lastword_top_beam_size_logprobs = which_sequence_top_beam_size_word_logprobs.gather(dim=-1, index=which_word)

            new_loop_dec_classes = torch.cat((new_loop_dec_classes, lastword_top_beam_size_classes), dim=-1)
            new_loop_dec_logprobs = torch.cat((new_loop_dec_logprobs, lastword_top_beam_size_logprobs), dim=-1)
            loop_dec_classes = new_loop_dec_classes
            loop_dec_logprobs = new_loop_dec_logprobs

            loop_cumul_logprobs = loop_dec_logprobs.sum(dim=-1, keepdims=True)

            loop_num_elem_vector = loop_num_elem_vector.view(bs, beam_size)[[bs_idxes, which_sequence]].view(bs * beam_size)
            there_was_eos_mask = (loop_dec_classes[:, :, :-1].view(bs, beam_size, time_step) == eos_idx). \
                sum(dim=-1).type(torch.bool).view(bs * beam_size)
            loop_num_elem_vector = loop_num_elem_vector + (1 * (1 - there_was_eos_mask.type(torch.int)))

            if (loop_num_elem_vector != time_step + 1).sum() == (bs * beam_size):
                break

        loop_cumul_logprobs /= loop_num_elem_vector.reshape(bs, beam_size, 1)
        _, topi = torch.topk(loop_cumul_logprobs.squeeze(-1), k=beam_size)
        res_caption_pred = [[] for _ in range(bs)]
        res_caption_logprob = [[] for _ in range(bs)]
        for i in range(bs):
            for j in range(how_many_outputs):
                idx = topi[i, j].item()
                res_caption_pred[i].append(
                    loop_dec_classes[i, idx, :loop_num_elem_vector[i * beam_size + idx]].tolist())
                res_caption_logprob[i].append(loop_dec_logprobs[i, idx, :loop_num_elem_vector[i * beam_size + idx]])

        flatted_res_caption_logprob = [logprobs for i in range(bs) for logprobs in res_caption_logprob[i]]
        flatted_res_caption_logprob = torch.nn.utils.rnn.pad_sequence(flatted_res_caption_logprob, batch_first=True)
        res_caption_logprob = flatted_res_caption_logprob.view(bs, how_many_outputs, -1)

        return res_caption_pred, res_caption_logprob
