import os
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from models.model import Zipformer
from utils.dataset import Speech2Text, speech_collate_fn
from jiwer import wer, cer
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ids_to_text(ids, itos, eos_id=None):
    tokens = []
    for idx in ids:
        if eos_id is not None and idx == eos_id:
            break
        token = itos.get(idx, '')
        if token in ['<pad>','<s>','</s>','<unk>','<blank>']:
            continue
        tokens.append(token)
    return ' '.join(tokens)

def main():
    parser = argparse.ArgumentParser(description="Inference script for RNN-T speech-to-text model")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--epoch', type=int, default=1, help='Epoch number to load the model from')
    parser.add_argument('--result', action='store_true',help='If set, save inference results to file, otherwise only print')
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    model_cfg = full_cfg.get('model', full_cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===Load Checkpoint===
    epoch = args.epoch
    checkpoint = torch.load(full_cfg["training"]["save_path"] + full_cfg["model"]["name"] + f"_epoch_{epoch}", map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    dataset = Speech2Text(full_cfg["training"]["train_path"], full_cfg["training"]["vocab_path"])
    vocab_size = len(dataset.vocab)

    #===Load Model===
    model = Zipformer(model_cfg, vocab_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #===Load Data===
    dataset = Speech2Text(full_cfg["training"]["test_path"], full_cfg["training"]["vocab_path"])
    itos    = dataset.vocab.itos
    eos_id  = dataset.vocab.get_eos_token()

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=speech_collate_fn)

    pred_texts = []
    true_texts = []

    # Nếu --result được bật thì mở file, ngược lại chỉ in ra màn hình
    fout = None
    if args.result:
        fout = open(full_cfg["training"]["result"], 'w', encoding='utf-8')

    for batch in loader:
        fbanks     = batch['fbank'].to(device)
        fbank_lens = batch['fbank_len'].to(device)
        speech_mask = batch["fbank_mask"].to(device)

        with torch.no_grad():
            batch_preds = model.recognize(fbanks, speech_mask)

        for i in range(len(batch_preds)):
            pred_ids = batch_preds[i]
            true_ids = batch['text'][i].tolist()

            pred_text = ids_to_text(pred_ids, itos, eos_id=eos_id)
            true_text = ids_to_text(true_ids, itos, eos_id=eos_id)

            pred_texts.append(pred_text)
            true_texts.append(true_text)

            print(f"Predict text: {pred_text}")
            print(f"Ground truth: {true_text}")

            if fout:
                fout.write(f"Predict text: {pred_text}\n")
                fout.write(f"Ground truth: {true_text}\n")
                fout.write("---------------\n")

    print("Inference complete.")

    #===TÍNH WER VÀ CER===
    overall_wer = wer(true_texts, pred_texts)
    overall_cer = cer(true_texts, pred_texts)

    print(f"Word Error Rate (WER): {overall_wer:.4f}")
    print(f"Character Error Rate (CER): {overall_cer:.4f}")

    if fout:
        fout.write(f"========== Tổng Kết ==========\n")
        fout.write(f"Word Error Rate (WER): {overall_wer:.4f}\n")
        fout.write(f"Character Error Rate (CER): {overall_cer:.4f}\n")
        fout.close()

if __name__ == '__main__':
    main()