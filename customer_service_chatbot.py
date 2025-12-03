import time
from typing import List

import pandas as pd
import torch
import matplotlib.pyplot as plt
# TODO: comment out on non-Windows devices
# import torch_directml 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import sys

from Vocab import Vocab
from GPTStyleTransformerLM import GPTStyleTransformerLM
from Seq2SeqTransformer import Seq2SeqTransformer
from GPTStyleDataset import GPTStyleDataset
from Seq2SeqDataset import Seq2SeqDataset

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
        DEVICE = torch.device("cpu")
        print("Not using gpu")
except ImportError:
    DEVICE = torch.device("cpu")
    print("No GPU backend found (CUDA/DirectML). Using CPU.")

BATCH_SIZE = 64
MAX_SEQ_LEN = 200
MAX_SRC_LEN = 100
MAX_TGT_LEN = 100
MIN_FREQ = 2

D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
DIM_FF = 512
DROPOUT = 0.1
NUM_EPOCHS = 50
LR = 1e-4

def pad_batch(seqs, pad_idx):
    """
    Returns batch with all sequences being the same length, with shorter sequences using <PAD> tokens to fill in the space.
    """
    # list of sequence lengths
    lengths = [len(s) for s in seqs]
    # maximum sequence length in the batch
    max_len = max(lengths)
    # fill a 2d tensor with pad_idxes
    out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long) # shape (batch size, max seq len)
    for i, s in enumerate(seqs):
        # iterate through each sequence in the batch and copy over sequence to padded; if shorter than max length, the rest of the tokens are pad
        out[i, :len(s)] = s
    return out

def collate_fn(batch, pad_idx):
    return pad_batch(batch, pad_idx)

def seq2seq_collate_fn(batch, pad_idx):
    src_seqs, tgt_seqs = zip(*batch)  # each is a 1D tensor

    src_batch = pad_batch(src_seqs, pad_idx)  # (batch, src_len)
    tgt_batch = pad_batch(tgt_seqs, pad_idx)  # (batch, tgt_len)

    return src_batch, tgt_batch


def train_epoch(model, dataloader, optimizer, criterion, pad_idx: int):
    """
    Runs one epoch of training for GPT-Style Model:
        1. Keep a running total loss and total non-pad tokens over all batches
        2. Iterate through each minibatch and:
            1. Get input and target tokens
            2. Forward pass inputs through model to get logit scores.
            3. Calculate loss using logit scores.
            4. Weight loss by number of non-pad tokens and add to total loss.
            5. Add number of non-pad tokens to total tokens.
        3. Return total loss / total non-pad tokens
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    # Iterate through each minibatch
    for x in dataloader:
        x = x.to(DEVICE) # shape (batch_size, max seq len from batch)

        # original sequence exmaple: [<bos>, hello, there, <pad>, <eos>]
        # input example: [<bos>, hello, there, <pad>]
        input_ids = x[:, :-1] # shape (batch_size, max seq len-1)
        # output example: [hello, there, <pad>, <eos>]
        target_ids = x[:, 1:] # shape (batch_size, max seq len-1)

        optimizer.zero_grad()
        # forward pass the inputs
        logits = model(input_ids) # shape (batch_size, max seq len-1, vocab size)

        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size), # shape (batch_size * max seq len - 1, vocab size)
            target_ids.reshape(-1) # shape (vocab_size, )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # for stability
        optimizer.step()

        # get number of non-pad tokens
        non_pad = (target_ids != pad_idx) 
        num_tokens = non_pad.sum().item()

        # get avg batch loss weighted by number of tokens
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    # total batch losses / number of non-pad tokens
    return total_loss / max(total_tokens, 1)


def evaluate(model, dataloader, criterion, pad_idx: int):
    """
    Calculates loss weighted by number of non-pad tokens for validation set for GPT-Style Model.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Iterate through minibatch
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


def train_epoch_seq2seq(model, dataloader, optimizer, criterion, pad_idx: int):
    """
    Runs one epoch of training for Seq2Seq Style Model.
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in dataloader:
        src = src.to(DEVICE)  # (batch, src_len)
        tgt = tgt.to(DEVICE)  # (batch, tgt_len)

        # Decoder input and target (teacher forcing)
        tgt_input  = tgt[:, :-1]  # includes <bos>, excludes final token
        tgt_output = tgt[:, 1:]   # what we want to predict

        optimizer.zero_grad()

        logits = model(src, tgt_input)  # (batch, tgt_len-1, vocab_size)

        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_output.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        non_pad = (tgt_output != pad_idx)
        num_tokens = non_pad.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def evaluate_seq2seq(model, dataloader, criterion, pad_idx: int):
    """
    Calculates loss weighted by number of non-pad tokens for validation set for GPT-Style Model.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            vocab_size = logits.size(-1)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_output.reshape(-1),
            )

            non_pad = (tgt_output != pad_idx)
            num_tokens = non_pad.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def generate_response(model, vocab, instruction, max_new_tokens = 500):
    model.eval()

    # Tokenize instruction
    instr_ids = vocab.numericalize(instruction, add_bos_eos=False)
    input_ids = [vocab.bos_idx()] + instr_ids + [vocab.sep_idx()]

    # TODO: dont think we need
    # if len(input_ids) >= MAX_SEQ_LEN - 1:
    #     input_ids = input_ids[:MAX_SEQ_LEN - 1]

    x = torch.tensor(input_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for step in range(max_new_tokens):
        # TODO: dont think we need
        # if x.size(1) >= MAX_SEQ_LEN:
        #     break

        logits = model(x)
        next_token_logits = logits[0, -1, :]

        if step == 0:
            # prevent generating EOS token right after instruction
            next_token_logits[vocab.eos_idx()] = float("-inf")

        # greedily select next token with highest logit (most probable)
        next_token_id = torch.argmax(next_token_logits).item()

        # append next token to input
        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=DEVICE)
        x = torch.cat([x, next_token], dim=1)

        if next_token_id == vocab.eos_idx():
            break

    generated_ids = x[0].tolist()

    # TODO: dont think we need
    # if vocab.sep_idx() in generated_ids:
    #     # split at <SEP> token and return only the response part
    #     sep_pos = generated_ids.index(vocab.sep_idx())
    #     response_ids = generated_ids[sep_pos + 1:]
    # else:
    #     # no <SEP> found, return everything after <BOS>
    #     response_ids = generated_ids[1:]

    # split at <SEP> token and return only the response part
    sep_pos = generated_ids.index(vocab.sep_idx())
    response_ids = generated_ids[sep_pos + 1:]

    text = vocab.denumericalize(response_ids)

    if not text.strip():
        text = "[no response generated]"

    return text


def generate_response_seq2seq(model, vocab, instruction, max_new_tokens=100):
    model.eval()

    # 1) Encode the instruction as src
    src_ids = vocab.numericalize(instruction, add_bos_eos=False, max_len=MAX_SEQ_LEN)
    src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, src_len)

    # 2) Start tgt with BOS
    tgt_ids = [vocab.bos_idx()]
    tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # (1, 1)

    for _ in range(max_new_tokens):
        # Call full forward each step (inefficient but simple)
        logits = model(src, tgt)                # (1, tgt_len, vocab_size)
        next_token_logits = logits[0, -1, :]    # last timestep

        next_token_id = torch.argmax(next_token_logits).item()
        tgt_ids.append(next_token_id)

        if next_token_id == vocab.eos_idx():
            break

        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    # Drop BOS, stop at EOS
    decoded_ids = tgt_ids[1:]
    if vocab.eos_idx() in decoded_ids:
        eos_pos = decoded_ids.index(vocab.eos_idx())
        decoded_ids = decoded_ids[:eos_pos]

    text = vocab.denumericalize(decoded_ids)
    if not text.strip():
        text = "[no response generated]"
    return text


def main(model_type):
    """
    Runs the following steps:
        1. Loads the dataset from HF
        2. Builds the Vocabulary
        3. Get training and validation splits
        4. Build model (either Seq2Seq or GPTStyleTransformer)
        5. Runs training for NUM_EPOCHS
        6. Save learning curves
    """
    print("Loading dataset from Hugging Face...")
    df = pd.read_csv("hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    # Keep only the two columns model expects
    df = df[["instruction", "response"]].dropna().reset_index(drop=True)
    print("Building vocabulary...")
    all_texts = df["instruction"].astype(str).tolist() + df["response"].astype(str).tolist()
    vocab = Vocab(min_freq=MIN_FREQ)
    vocab.build(all_texts)
    print(f"Vocab size: {len(vocab.itos)}")

    # GPT-STYLE MODEL
    if model_type == 0:
        # Create GPT-Style Dataset (combine instruction and response) and split into training and validation sets
        dataset = GPTStyleDataset(df, vocab=vocab, max_len=MAX_SEQ_LEN)
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        # Create train/validation DataLoader objects, use collate_fn to pad sequences in a batch
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

        # Create model
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
        train_fn = train_epoch
        eval_fn  = evaluate

    # SEQ2SEQ STYLE MODEL
    else: 
        dataset = Seq2SeqDataset(df, vocab=vocab, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN)
        val_size = int(0.1 * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=lambda b: seq2seq_collate_fn(b, vocab.pad_idx()),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=lambda b: seq2seq_collate_fn(b, vocab.pad_idx()),
        )

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

        train_fn = train_epoch_seq2seq
        eval_fn  = evaluate_seq2seq

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss = train_fn(model, train_loader, optimizer, criterion, vocab.pad_idx())
        val_loss = eval_fn(model, val_loader, criterion, vocab.pad_idx())
        elapsed = time.time() - start
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # If validation loss improved, save current model to .pt file
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

    print("Creating learning curve plots...")
    fig, ax = plt.subplots()
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss")
    ax.plot(epochs, val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"{path[:-3]}_learning_curves.png")
    print(f"Saved learning curves to {path[:-3]}_learning_curves.png.")

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


def chat(model, model_type, vocab, max_new_tokens=50):
    print("Customer Service Chatbot — type 'exit' to quit.")
    while True:
        user_input = input("Customer: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if model_type == 1:
            response_fn = generate_response_seq2seq
        else:
            response_fn = generate_response
        reply = response_fn(model, vocab, user_input, max_new_tokens=max_new_tokens)
        print(f"Agent: {reply}")


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "chat"
    model_type = 1 if len(sys.argv) > 2 and sys.argv[2] == "1" else 0

    if mode == "train":
        main(model_type)
    elif mode == "chat":
        path = "seq2seq_customer_service_bot.pt" if model_type == 1 else "gpt_style_customer_service_bot.pt"
        model, vocab = load_model_for_inference(path, model_type)
        chat(model, model_type, vocab, max_new_tokens=750)
    else:
        print("Usage: python customer_service_chatbot.py [chat|train] [model_type]")
