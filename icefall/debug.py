import torch
from zip_utils.dataset import Speech2Text, speech_collate_fn, compute_gmvn
from zip_models.model import Transducer
from zip_models.loss import RNNTLoss
import argparse
import yaml
import os
from zip_models.optim import Optimizer

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def debug_batch(model, batch, criterion, device, is_training=True):
    mode = "Training" if is_training else "Evaluating"
    print(f"\n=== Debugging in {mode} Batch ===")
    
    # Move batch to device
    speech = batch["fbank"].to(device)
    speech_mask = batch["fbank_mask"].to(device)
    text_mask = batch["text_mask"].to(device)
    fbank_len = batch["fbank_len"].to(device)
    text_len = batch["text_len"].to(device)
    target_text = batch["text"].to(device)
    decoder_input = batch["decoder_input"].to(device)

    # Print shapes
    print(f"================INPUT================")
    print(f"Speech shape: {speech.shape}")
    print(f"Speech mask shape: {speech_mask.shape}")
    print(f"Text mask shape: {text_mask.shape}")
    print(f"Fbank len shape: {fbank_len.shape}")
    print(f"Text len shape: {text_len.shape}")
    print(f"Target text shape: {target_text.shape}")
    print(f"Text sample: {target_text[0, :10]}")  # Print first 10 tokens
    print(f"Decoder input shape: {decoder_input.shape}")
    print(f"Decoder input sample: {decoder_input[0, :10]}")  # Print first 10 tokens
    print(f"================MODEL================")
    # Set model mode
    if is_training:
        model.train()
    else:
        model.eval()

    # Forward pass
    output = model(speech, fbank_len.long(), decoder_input.int(), text_len.long())
    
    print(f"Model output shape: {output.shape}")
    print(f"Sample model output values: {output[0, :3, :3]}")  # Print first few values

    # Compute loss
    loss = criterion(output, target_text, fbank_len, text_len)
    print(f"Loss value: {loss.item()}")

    # if is_training:
    #     # Backward pass
    #     loss.backward()
    #     print(f"Sample gradient for a model parameter:")
    #     for name, param in model.named_parameters():
    #         if param.grad is not None:
    #             print(f"  {name} grad shape: {param.grad.shape}")
    #             print(f"  {name} grad sample: {param.grad.flatten()[:5]}")
    #             break  # Print only one parameter's gradient for brevity

    return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']

    # ==== Compute gmvn ====
    # gmvn_mean, gmvn_std = compute_gmvn(training_cfg['voice_path'])

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
        # gmvn_mean = gmvn_mean,
        # gmvn_std = gmvn_std,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        collate_fn=speech_collate_fn
    )

    # ==== Model ====
    model = Transducer(config['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ==== Loss ====
    criterion = RNNTLoss(config["rnnt_loss"]["blank"], config["rnnt_loss"]["reduction"])

    # ==== Get one batch ====
    # Get a batch from the training set to debug
    train_batch = next(iter(train_loader))

    # ==== Debug training batch ====
    train_loss = debug_batch(model, train_batch, criterion, device, is_training=True)
    print(f"✅ Training batch loss: {train_loss:.4f}")

    # # ==== Debug evaluation batch ====
    # # Get a batch from the validation set to debug
    # with torch.no_grad():
    #     val_loss = debug_batch(model, dev_batch, criterion, device, is_training=False)
    # print(f"✅ Validation batch loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()