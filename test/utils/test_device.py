from unittest.mock import patch

from haystack.utils import get_device


@patch("torch.cuda.is_available")
def test_get_device_cuda(torch_cuda_is_available):
    torch_cuda_is_available.return_value = True
    device = get_device()
    assert device == "cuda:0"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_get_device_mps(torch_cuda_is_available, torch_backends_mps_is_available):
    torch_cuda_is_available.return_value = False
    torch_backends_mps_is_available.return_value = True
    device = get_device()
    assert device == "mps:0"


@patch("torch.backends.mps.is_available")
@patch("torch.cuda.is_available")
def test_get_device_cpu(torch_cuda_is_available, torch_backends_mps_is_available):
    torch_cuda_is_available.return_value = False
    torch_backends_mps_is_available.return_value = False
    device = get_device()
    assert device == "cpu:0"
