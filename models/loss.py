from torchaudio.functional import rnnt_loss
import torch

class RNNTLoss(torch.nn.Module):
    def __init__(self, blank=0, reduction="mean"):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction

    def forward(self, logits, targets, fbank_len, text_len):
        # logits: [B, T, U, vocab_size]
        # targets: [B, U]
        # fbank_len: [B]
        # text_len: [B]

        loss = rnnt_loss(logits, targets[:, :].int(), fbank_len.int(), text_len.int(), blank=self.blank)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss