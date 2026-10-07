from src.preprocess import preprocess_text, prepare_dataset


def test_preprocess_text():
    assert preprocess_text("Hello World!") == "hello world"
    assert preprocess_text("MURaho, nagufasha?") == "muraho nagufasha"
    assert preprocess_text("") == ""


def test_prepare_dataset():
    batch = {
        "text": "Test text.",
        "audio": {
            "array": [0.0] * 32000,
            "sampling_rate": 16000
        }
    }

    # Duration is 2.0s
    res = prepare_dataset(batch, min_duration=1.0, max_duration=3.0)
    assert res["is_valid"] is True
    assert res["text_normalized"] == "test text"

    # Out of bounds
    res2 = prepare_dataset(batch, min_duration=3.0, max_duration=5.0)
    assert res2["is_valid"] is False
