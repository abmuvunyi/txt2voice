import os
from datasets import load_dataset, Audio, DatasetDict, IterableDataset
from src.preprocess import filter_dataset

def get_tts_dataset(config):
    """
    Loads the dataset based on configuration.
    """
    dataset_name = config.get("dataset_name", "Washere-1/tecgrw-audio")
    use_proxy = config.get("use_proxy", False)
    sampling_rate = config.get("sampling_rate", 16000)

    if use_proxy:
        print(f"Loading PROXY dataset: hf-internal-testing/librispeech_asr_dummy (clean, validation)...")
        # specific subset
        # Remove trust_remote_code=True
        dataset_stream = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", streaming=True)
        
        print("Streaming all samples (it is small)...")
        dataset_head = list(dataset_stream)
        
        from datasets import Dataset
        dataset = Dataset.from_list(dataset_head)
        
        # Add dummy speaker_id if not present
        if "speaker_id" not in dataset.column_names:
             dataset = dataset.add_column("speaker_id", [0] * len(dataset))
        
        # Split into train/val
        if len(dataset) > 1:
            dataset = dataset.train_test_split(test_size=0.2)
        else:
            # Handle tiny dataset edge case
            dataset = DatasetDict({"train": dataset, "test": dataset})
        
    else:
        print(f"Loading REAL dataset: {dataset_name}...")
        try:
            dataset = load_dataset(dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {dataset_name}. Ensure you have access rights. Error: {e}")

    # Standardize columns
    # librispeech dummy: audio, text, file, id, speaker_id, chapter_id
    columns_to_keep = ["audio", "text", "speaker_id"]
    
    dataset = dataset.select_columns(columns_to_keep)
    
    # Cast audio column to desired sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    return dataset

if __name__ == "__main__":
    # Test stub
    conf = {"use_proxy": True, "sampling_rate": 16000}
    ds = get_tts_dataset(conf)
    print("Dataset structure:", ds)
    print("Sample:", ds["train"][0])
    
    # Verify preprocessing
    ds_processed = filter_dataset(ds["train"], min_duration=0.1) # low min_duration for dummy
    print("Processed sample:", ds_processed[0])
