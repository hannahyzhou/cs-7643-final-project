import sys
import random

import pandas as pd

from bert_score import score as bertscore_score

from customer_service_chatbot import (
    load_model_for_inference,
    generate_response,
    DATA_PATH,
)


def evaluate_bertscore(max_new_tokens=750):
    data_path = DATA_PATH

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    df = df[["instruction", "response"]].dropna().reset_index(drop=True)
    df = df.head(int(len(df) * 0.05))

    checkpoint_path = "gpt_style_customer_service_bot.pt"

    print(f"Loading model from: {checkpoint_path}")
    model, vocab = load_model_for_inference(checkpoint_path)

    print("Generating responses...")
    refs = []
    hyps = []

    for i, row in df.iterrows():
        instruction = str(row["instruction"])
        true_response = str(row["response"])

        generated_response = generate_response(model, vocab, instruction, max_new_tokens=max_new_tokens)

        refs.append(true_response)
        hyps.append(generated_response)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(df)} examples...")

    print("Computing BERTScore (this may take a bit)...")
    P, R, F1 = bertscore_score(hyps, refs, lang="en", rescale_with_baseline=True)

    avg_P = P.mean().item()
    avg_R = R.mean().item()
    avg_F1 = F1.mean().item()

    print("\n===== BERTScore Results =====")
    print(f"Average Precision: {avg_P:.4f}")
    print(f"Average Recall:    {avg_R:.4f}")
    print(f"Average F1:        {avg_F1:.4f}")
    print("=============================\n")


if __name__ == "__main__":
    evaluate_bertscore(750)