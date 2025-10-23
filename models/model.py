import torch
from torch import nn
import sys
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()
        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs
    
class Zipformer(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config, vocab_size)
        self.joint_net = JointNet(
            input_size=config['joint']['input_size'],
            inner_dim=config['joint']['hidden_size'],
            vocab_size=vocab_size
        )

    def forward(self, speech, mask, dec_input, target_lens):
        encoder_out, encoder_out_lens = self.encode(speech, mask)

        decoder_out, _ = self.decode(dec_input, target_lens)

        joint_out = self.joint_net(encoder_out, decoder_out)  # [B, T, U, vocab_size]

        return joint_out, encoder_out_lens
    
    def encode(self, x, mask):
        encoder_out, encoder_out_lens = self.encoder(x, mask)

        return encoder_out, encoder_out_lens
    
    def decode(self, dec_input, target_lens= None, hidden= None):
        decoder, hid = self.decoder(dec_input, target_lens, hidden)
        
        return decoder, hid
    
    def recognize(self, speech_feature, mask):
        batch_size = speech_feature.size(0)
        enc_states, enc_mask, enc_lens = self.encode(speech_feature, mask, None)  # [B, T, out_dim]

        zero_token = torch.LongTensor([[1]])
        if speech_feature.is_cuda:
            zero_token = zero_token.cuda()

        def infer(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decode(zero_token)

            for t in range(lengths):
                logits = self.joint_net(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0).item()

                if pred == 2: 
                    break

                if pred not in (0, 1, 2, 4) and (len(token_list) == 0 or pred != token_list[-1]):
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list
        
        results = [infer(enc_states[i], enc_lens[i]) for i in range(batch_size)]

        return results
