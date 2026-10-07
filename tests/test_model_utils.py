from unittest.mock import patch
from src.model_utils import get_device, load_tts_model


def test_get_device_force_cpu():
    assert get_device(force_cpu=True) == "cpu"


@patch('torch.cuda.is_available', return_value=True)
def test_get_device_cuda(mock_cuda):
    assert get_device(force_cpu=False) == "cuda"


@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.mps.is_available', return_value=True)
def test_get_device_mps(mock_mps, mock_cuda):
    assert get_device(force_cpu=False) == "mps"


@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.mps.is_available', return_value=False)
def test_get_device_cpu_fallback(mock_mps, mock_cuda):
    assert get_device(force_cpu=False) == "cpu"


@patch('src.model_utils.VitsModel.from_pretrained')
@patch('src.model_utils.VitsTokenizer.from_pretrained')
def test_load_tts_model_success(mock_tokenizer, mock_model):
    from unittest.mock import MagicMock
    mock_model_instance = MagicMock()
    mock_model_instance_to = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance_to
    mock_model.return_value = mock_model_instance
    model, tokenizer, device, model_id = load_tts_model(
        model_path="dummy_model", device="cpu")
    mock_model.assert_called_with("dummy_model")
    mock_tokenizer.assert_called_with("dummy_model")
    assert device == "cpu"
    assert model_id == "dummy_model"


@patch('src.model_utils.VitsModel.from_pretrained')
@patch('src.model_utils.VitsTokenizer.from_pretrained')
def test_load_tts_model_fallback(mock_tokenizer, mock_model):
    # Mocking the model to have a .to() method
    from unittest.mock import MagicMock
    mock_model_instance = MagicMock()
    mock_model_instance_to = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance_to

    # Make the first call fail
    mock_model.side_effect = [Exception("Failed"), mock_model_instance]

    model, tokenizer, device, model_id = load_tts_model(
        model_path="dummy_model", base_model="base_dummy", device="cpu")

    assert mock_model.call_count == 2
    mock_model.assert_any_call("dummy_model")
    mock_model.assert_any_call("base_dummy")
    assert model_id == "base_dummy"
