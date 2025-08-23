from __future__ import annotations

import torch
from torch import nn
import sys
import torch.nn.functional as F
from egs.librispeech.ASR.zipformer.zipformer import Zipformer2
from egs.librispeech.ASR.zipformer.subsampling import Conv2dSubsampling

from .decoder import build_decoder
from zip_utils.dataset import calculate_mask, causal_mask

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
    def __init__(self, config):
        super().__init__()
           
        self.conv_embeded = Conv2dSubsampling(
            in_channels=config['conv_embeded']['in_channels'],
            out_channels=config['conv_embeded']['out_channels'],
            layer1_channels=config['conv_embeded']['layer1_channels'],
            layer2_channels=config['conv_embeded']['layer2_channels'],
            layer3_channels=config['conv_embeded']['layer3_channels'],
            dropout=config['conv_embeded']['dropout'],
        )

        self.encoder = Zipformer2(
            output_downsampling_factor= 2, # keep it intact as code demands
            downsampling_factor= (1,1,1,1,1,1),  # dont know
            encoder_dim= (192,256,256,256,256,256),  # paper
            num_encoder_layers= (2, 2, 2, 2, 2, 2),  # each stage has x2 zipformer blocks
            encoder_unmasked_dim= (192,256,256,256,256,256),  # recommendeded in code, not really sure it had in paper
            query_head_dim= 32,     # paper
            pos_head_dim= 4,
            value_head_dim= 12,     # paper
            num_heads= (4,4,4,8,4,4),   # paper
            feedforward_dim= (512,768,768,768,768,768),  # paper
            cnn_module_kernel= (31,31,15,15,15,31),     # paper
            pos_dim= 192,
            dropout= 0.1,  # maybe 
            warmup_batches= 8000.0, # maybe
            causal= True, # maybe 
            chunk_size= [16],  # ?
            left_context_frames= [64],     # maybe
        )

        self.decoder = build_decoder(config)

        self.joint_net = JointNet(
            input_size=config['joint']['input_size'],
            inner_dim=config['joint']['hidden_size'],
            vocab_size=config['vocab_size']
        )

    def forward(self, speech, fbank_lens, dec_input, target_lens, use_mask:bool= None):
        encoder_out, encoder_out_lens = self.encode(speech, fbank_lens, use_mask)

        decoder_out, hidden = self.decode(dec_input, target_lens)

        joint_out = self.joint_net(encoder_out, decoder_out)  # [B, T, U, vocab_size]

        return joint_out, encoder_out_lens
    
    def encode(self, x: torch.Tensor, x_lens: torch.Tensor, use_mask:bool= None):
        x, x_lens = self.conv_embeded(x, x_lens) # [batch, time, features] 
        
        if use_mask:
            max_subsampled_len = x.size(1)
            mask = ~calculate_mask(x_lens, max_subsampled_len)
            
        x = x.transpose(0, 1)  # [batch, time, features] -> (time, batch, feature)
        out = self.encoder(x, x_lens, mask)
        encoder_out, encoder_out_lens = out[0], out[1]
        encoder_out = encoder_out.transpose(0, 1)

        return encoder_out, encoder_out_lens
    
    def decode(self, dec_input, target_lens= None, hidden= None):
        decoder, hid = self.decoder(dec_input, target_lens, hidden)
        
        return decoder, hid
    
    def recognize(self, speech_feature, fbank_lens, use_mask:bool= True):
        batch_size = speech_feature.size(0)
        enc_states, _ = self.encode(speech_feature, fbank_lens, use_mask)  # [B, T, out_dim]

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

                print(pred)
                if pred == 2: 
                    break

                if pred not in (0, 1, 2, 4) and (len(token_list) == 0 or pred != token_list[-1]):
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list
        
        results = [infer(enc_states[i], fbank_lens[i]) for i in range(batch_size)]

        return results

