
import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing_coeff, rank='cuda:0'):
        assert 0.0 <= smoothing_coeff <= 1.0
        super().__init__()
        self.smoothing_coeff = smoothing_coeff
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.rank = rank

    def forward(self, pred, target, ignore_index, divide_by_non_zeros=True):
        pred = self.log_softmax(pred)

        batch_size, seq_len, num_classes = pred.shape
        uniform_confidence = self.smoothing_coeff / (num_classes - 1)  # minus one cause of PAD token
        confidence = 1 - self.smoothing_coeff
        one_hot = torch.full((num_classes,), uniform_confidence).to(self.rank)
        model_prob = one_hot.repeat(batch_size, seq_len, 1)
        model_prob.scatter_(2, target.unsqueeze(2), confidence)
        model_prob.masked_fill_((target == ignore_index).unsqueeze(2), 0)

        tot_loss_tensor = self.kl_div(pred, model_prob)

        # divide the loss of each sequence by the number of non pads
        pads_matrix = torch.as_tensor(target == ignore_index)
        tot_loss_tensor.masked_fill_(pads_matrix.unsqueeze(2), 0.0)
        if divide_by_non_zeros:
            num_non_pads = (~pads_matrix).sum().type(torch.cuda.FloatTensor)
            tot_loss = tot_loss_tensor.sum() / num_non_pads
        else:
            tot_loss = tot_loss_tensor.sum()

        return tot_loss
