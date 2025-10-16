import torch
from tqdm import tqdm
import argparse
import yaml
import os 
from torch import nn
from speechbrain.nnet.schedulers import NoamScheduler
import warnings
from utils import Speech2Text, speech_collate_fn, calculate_mask, causal_mask
from model.encoder import ConvEmbeded, ZipformerEncoder

warnings.filterwarnings("ignore", category=UserWarning)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']

    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        collate_fn=speech_collate_fn
    )

    vocab_size = len(train_dataset.vocab)

    model_cfg = config['model']
    conv_embeded = ConvEmbeded(
        input_dim=model_cfg['conv_embeded']['input_dim'],
        output_dim=model_cfg['conv_embeded']['output_dim'],
        conv_dim=model_cfg['conv_embeded']['conv_dim'],
    )

    for batch in tqdm(train_loader, desc="🔁 Training", leave=False): 
        speech = batch["fbank"]
        speech_mask = batch["fbank_mask"]
        text_mask = batch["text_mask"]
        fbank_len = batch["fbank_len"]
        text_len = batch["text_len"]
        target_text = batch["text"]
        decoder_input = batch["decoder_input"]
        tokens = batch["tokens"]
        tokens_lens = batch["tokens_lens"]

        # conv_out = conv_embeded(speech)
        # new_lengths = torch.tensor([
        #     conv_embeded.calculate_output_length(length.item()) 
        #     for length in fbank_len
        # ])
        # new_mask = calculate_mask(new_lengths, conv_out.size(1))  # (B, T')

        output, new_mask = ZipformerEncoder(model_cfg, vocab_size).forward(speech, fbank_len, speech_mask)

        print(output.shape, speech_mask.shape, new_mask.shape)
        exit()
        


if __name__ == "__main__":
    main()
