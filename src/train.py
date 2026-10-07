import argparse
import torch
from transformers import VitsModel, VitsTokenizer, TrainingArguments, Trainer
from src.dataset import get_tts_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/mms-tts-eng")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--use_proxy",
        action="store_true",
        help="Use proxy dataset (LibriTTS)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # 1. Load Dataset
    config = {
        "use_proxy": args.use_proxy,
        "dataset_name": "librispeech_asr" if args.use_proxy else "Washere-1/tecgrw-audio"}
    dataset = get_tts_dataset(config)

    # 2. Load Model & Tokenizer
    model = VitsModel.from_pretrained(args.model_name)  # nosec B615
    tokenizer = VitsTokenizer.from_pretrained(args.model_name)  # nosec B615

    # 3. Data Collator
    def collate_fn(batch):
        # Extract inputs
        input_ids = [
            tokenizer(
                x["text"],
                return_tensors="pt").input_ids[0] for x in batch]

        # Pad inputs using torch.nn.utils.rnn.pad_sequence for performance
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

        # Create attention mask
        attention_mask = torch.zeros_like(input_ids_padded)
        for i, seq in enumerate(input_ids):
            attention_mask[i, :seq.size(0)] = 1

        # Extract audio (targets/labels)
        labels = [
            torch.tensor(
                x["audio"]["array"],
                dtype=torch.float32) for x in batch]
        # Pad labels (audio)
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=0.0)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",  # CHANGED from evaluation_strategy
        eval_steps=50,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
        data_collator=collate_fn,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
