import torch
from tqdm import tqdm
import argparse
import yaml
import os 
from torch import nn
from speechbrain.nnet.schedulers import NoamScheduler
import warnings
from utils.dataset import Speech2Text, speech_collate_fn
from model.encoder import ConvEmbeded

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

        conv_out = conv_embeded(speech)
        print(conv_out.shape)
        exit()
        


if __name__ == "__main__":
    main()
