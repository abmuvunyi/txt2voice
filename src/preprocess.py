import re


def preprocess_text(text):
    """
    Basic text normalization.
    For Kinyarwanda, we might need specific rules later.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation structure
    return text


def prepare_dataset(
        batch,
        tokenizer=None,
        feature_extractor=None,
        max_duration=12.0,
        min_duration=2.0):
    """
    Function to map over the dataset.
    """
    audio = batch["audio"]

    # Calculate duration
    duration = len(audio["array"]) / audio["sampling_rate"]
    batch["duration"] = duration

    # Validate Duration
    if duration < min_duration or duration > max_duration:
        batch["is_valid"] = False
    else:
        batch["is_valid"] = True

    # Normalize Text
    batch["text_normalized"] = preprocess_text(batch["text"])

    return batch


def filter_dataset(dataset, min_duration=2.0, max_duration=12.0):
    """
    Applies filtering and preprocessing.
    """
    print("Preprocessing and filtering dataset...")

    # Map preprocessing
    dataset = dataset.map(
        lambda x: prepare_dataset(
            x,
            min_duration=min_duration,
            max_duration=max_duration))

    # Filter invalid
    dataset = dataset.filter(lambda x: x["is_valid"])

    # Print stats
    print(f"Dataset size after filtering: {len(dataset)}")

    return dataset
