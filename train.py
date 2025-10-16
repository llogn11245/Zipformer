import torch
from tqdm import tqdm
import argparse
import yaml
import os 
import logging
from torch import nn
from speechbrain.nnet.schedulers import NoamScheduler
import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

def reload_model(model, optimizer, checkpoint_path, model_name):
    past_epoch = 0
    path_list = [path for path in os.listdir(checkpoint_path)]
    print(path_list)
    if len(path_list) > 0:
        for path in path_list:
            try:
                past_epoch = max(int(path.split("_")[-1]), past_epoch)
            except:
                continue
        
        load_path = os.path.join(checkpoint_path, f"{model_name}_epoch_{past_epoch}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Reloaded model from {load_path} at epoch {past_epoch}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")
    
    return past_epoch + 1, model, optimizer

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for batch in progress_bar:
        speech = batch["fbank"].to(device)
        speech_mask = batch["fbank_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        fbank_len = batch["fbank_len"].to(device)
        text_len = batch["text_len"].to(device)
        target_text = batch["text"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        tokens = batch["tokens"].to(device)
        tokens_lens = batch["tokens_lens"].to(device)
        optimizer.zero_grad()
        output, new_fbank_lens = model(speech, fbank_len.long(), decoder_input.int(), tokens_lens.cpu(), True)
        loss = criterion(output, tokens, new_fbank_lens, tokens_lens)
        loss.backward()
        

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=200)

        optimizer.step()

        lr , _ = scheduler(optimizer.optimizer)

        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average training loss: {avg_loss:.4f}")
    return avg_loss, lr


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            speech = batch["fbank"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            fbank_len = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)
            target_text = batch["text"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            tokens = batch["tokens"].to(device)
            tokens_lens = batch["tokens_lens"].to(device)

            output, new_fbank_lens = model(speech, fbank_len.long(), decoder_input.int(), tokens_lens.cpu(), True)
            loss = criterion(output, tokens, new_fbank_lens, tokens_lens)

            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Average validation loss: {avg_loss:.4f}")
    return avg_loss


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']

    # ==== Logger ====
    if not os.path.exists(training_cfg['log_path']):
        os.makedirs(os.path.dirname(training_cfg['log_path']), exist_ok=True)

        with open(training_cfg['log_path'], 'w') as f:
            pass  # Create the file if it doesn't exist

    log_file = training_cfg['log_path']
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )

    # ==== Load Dataset ====
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

    dev_dataset = Speech2Text(
        json_path=training_cfg['dev_path'],
        vocab_path=training_cfg['vocab_path'],
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        collate_fn=speech_collate_fn
    )

    vocab_size = len(train_dataset.vocab)

    # ==== Model ====
    model = Zipformer(config['model'], vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ==== Loss ====
    criterion = RNNTLoss(config["rnnt_loss"]["blank"], config["rnnt_loss"]["reduction"])

    # ==== Optimizer ====
    optimizer = Optimizer(model.parameters(), config['optim'])

    # ==== Scheduler ====
    if not config['training']['reload']:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
    else:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
        scheduler.load(config['training']['save_path'] + '/scheduler.ckpt')

    # ==== Reload checkpoint if needed ====
    start_epoch = 1
    if training_cfg['reload']:
        checkpoint_path = training_cfg['save_path']
        start_epoch, model, optimizer = reload_model(model, optimizer, checkpoint_path, config['model']['name'])

    # ==== Training loop ====
    num_epochs = training_cfg["epochs"]

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, lr = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = evaluate(model, dev_loader, criterion, device)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Lr = {lr:.6f}")

        # Save model checkpoint
        if not os.path.exists(training_cfg['save_path']):
            os.makedirs(training_cfg['save_path'])
        model_filename = os.path.join(
            training_cfg['save_path'],
            f"{config['model']['name']}_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        # Step scheduler with validation loss
        scheduler.save(os.path.join(training_cfg['save_path'], 'scheduler.ckpt'))


if __name__ == "__main__":
    main()
