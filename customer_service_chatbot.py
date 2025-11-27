import time
from typing import List

import pandas as pd
import torch
# TODO: comment out on non-Windows devices
# import torch_directml 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys

from Vocab import Vocab
from GPTStyleTransformerLM import GPTStyleTransformerLM
from Seq2SeqTransformer import Seq2SeqTransformer
from GPTStyleDataset import GPTStyleDataset

DATA_PATH = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

try:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        # TODO: comment out on non-Windows devices
        # import torch_directml
        # DEVICE = torch_directml.device()
        # print("Using DirectML device (AMD/Intel/Nvidia via DirectX).")
        print("Not using gpu")
except ImportError:
    DEVICE = torch.device("cpu")
    print("No GPU backend found (CUDA/DirectML). Using CPU.")

BATCH_SIZE = 64
MAX_SEQ_LEN = 200
MIN_FREQ = 2

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FF = 512
DROPOUT = 0.1

NUM_EPOCHS = 20
LR = 1e-3

def collate_fn(batch, pad_idx):
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)

    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq

    return padded


def train_epoch(model, dataloader, optimizer, criterion, pad_idx: int):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x in dataloader:
        x = x.to(DEVICE)

        input_ids = x[:, :-1]
        target_ids = x[:, 1:]

        optimizer.zero_grad()
        logits = model(input_ids)

        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        non_pad = (target_ids != pad_idx)
        num_tokens = non_pad.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def evaluate(model, dataloader, criterion, pad_idx: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x in dataloader:
        x = x.to(DEVICE)

        input_ids = x[:, :-1]
        target_ids = x[:, 1:]

        logits = model(input_ids)
        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1)
        )

        non_pad = (target_ids != pad_idx)
        num_tokens = non_pad.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)

def generate_response(model, vocab, instruction, max_new_tokens = 500):
    model.eval()

    instr_ids = vocab.numericalize(instruction, add_bos_eos=False)
    input_ids = [vocab.bos_idx()] + instr_ids + [vocab.sep_idx()]

    if len(input_ids) >= MAX_SEQ_LEN - 1:
        input_ids = input_ids[:MAX_SEQ_LEN - 1]

    x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for step in range(max_new_tokens):
        if x.size(1) >= MAX_SEQ_LEN:
            break

        logits = model(x)
        next_token_logits = logits[0, -1, :]

        if step == 0:
            next_token_logits[vocab.eos_idx()] = float("-inf")

        next_token_id = torch.argmax(next_token_logits).item()

        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
        x = torch.cat([x, next_token], dim=1)

        if next_token_id == vocab.eos_idx():
            break

    generated_ids = x[0].tolist()

    if vocab.sep_idx() in generated_ids:
        sep_pos = generated_ids.index(vocab.sep_idx())
        response_ids = generated_ids[sep_pos + 1:]
    else:
        response_ids = generated_ids[1:]
    text = vocab.denumericalize(response_ids)

    if not text.strip():
        text = "[no response generated]"

    return text

def main(model_type):
    print("Loading dataset from Hugging Face...")
    df = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    # Keep only the two columns your model expects
    df = df[["instruction", "response"]].dropna().reset_index(drop=True)


    print("Building vocabulary...")
    all_texts = df["instruction"].astype(str).tolist() + df["response"].astype(str).tolist()
    vocab = Vocab(min_freq=MIN_FREQ)
    vocab.build(all_texts)
    print(f"Vocab size: {len(vocab.itos)}")

    dataset = GPTStyleDataset(df, vocab=vocab, max_len=MAX_SEQ_LEN)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx()),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx()),
    )
    if model_type == 1:
        model = Seq2SeqTransformer(
            vocab_size=len(vocab.itos),
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=2,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            pad_idx=vocab.pad_idx(),
        ).to(DEVICE)
        path = "seq2seq_customer_service_bot.pt"
    else:
        model = GPTStyleTransformerLM(
            vocab_size=len(vocab.itos),
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            pad_idx=vocab.pad_idx(),
        ).to(DEVICE)
        path = "gpt_style_customer_service_bot.pt"

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, vocab.pad_idx())
        val_loss = evaluate(model, val_loader, criterion, vocab.pad_idx())
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab.itos,
                "config": {
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FF,
                    "dropout": DROPOUT,
                    "pad_idx": vocab.pad_idx(),
                    "max_seq_len": MAX_SEQ_LEN,
                }
            }, path)
            print("  -> Saved new best model.")

    print("Training complete.")

def load_model_for_inference(checkpoint_path, model_type):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    itos = ckpt["vocab"]
    vocab = Vocab()
    vocab.itos = itos
    vocab.stoi = {tok: i for i, tok in enumerate(itos)}

    config = ckpt["config"]

    if model_type == 1:
        model = Seq2SeqTransformer(
            vocab_size=len(vocab.itos),
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=2,
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            pad_idx=config["pad_idx"],
        ).to(DEVICE)
    else:
        model = GPTStyleTransformerLM(
            vocab_size=len(vocab.itos),
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            pad_idx=config["pad_idx"],
        ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


def chat(model, vocab, max_new_tokens= 50):
    print("Customer Service Chatbot — type 'exit' to quit.")
    while True:
        user_input = input("Customer: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        reply = generate_response(model, vocab, user_input, max_new_tokens=max_new_tokens)
        print(f"Agent: {reply}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "1":
        model_type = 1
    else:
        model_type = 0
    # Train the GPT-style LM
    # main(model_type)

    # # After training, you could do (in a separate script or REPL):
    if model_type == 1:
        path = "seq2seq_customer_service_bot.pt"
    else:
        path = "gpt_style_customer_service_bot.pt"
    model, vocab = load_model_for_inference(path, model_type)
    chat(model, vocab, max_new_tokens=750)
