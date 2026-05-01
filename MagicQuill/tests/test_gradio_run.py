import sys
import unittest.mock
import torch
import base64
import io
import pytest
from PIL import Image
import numpy as np

mock_modules = {
    'gradio': unittest.mock.MagicMock(),
    'gradio_magicquill': unittest.mock.MagicMock(),
    'fastapi': unittest.mock.MagicMock(),
    'uvicorn': unittest.mock.MagicMock(),
    'MagicQuill.folder_paths': unittest.mock.MagicMock(),
    'MagicQuill.llava_new': unittest.mock.MagicMock(),
    'MagicQuill.scribble_color_edit': unittest.mock.MagicMock(),
}

with unittest.mock.patch.dict(sys.modules, mock_modules):
    from gradio_run import tensor_to_base64, read_base64_image

def test_tensor_to_base64():
    tensor = torch.zeros((1, 10, 10, 3), dtype=torch.float32)
    base64_str = tensor_to_base64(tensor)

    assert isinstance(base64_str, str)
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    assert img.format == "PNG"
    assert img.size == (10, 10)

def test_read_base64_image_png():
    img = Image.new("RGB", (10, 10), color="red")
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    base64_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    read_img = read_base64_image(base64_str)
    assert read_img.size == (10, 10)
    assert np.array_equal(np.array(read_img), np.array(img))

def test_read_base64_image_jpeg():
    img = Image.new("RGB", (10, 10), color="blue")
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    base64_str = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    read_img = read_base64_image(base64_str)
    assert read_img.size == (10, 10)

def test_read_base64_image_webp():
    img = Image.new("RGB", (10, 10), color="green")
    buffered = io.BytesIO()
    img.save(buffered, format="WEBP")
    base64_str = "data:image/webp;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    read_img = read_base64_image(base64_str)
    assert read_img.size == (10, 10)

def test_read_base64_image_unsupported():
    with pytest.raises(ValueError, match="Unsupported image format."):
        read_base64_image("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

def test_round_trip():
    tensor = torch.ones((1, 20, 20, 3), dtype=torch.float32) * 0.5
    base64_str = tensor_to_base64(tensor)

    full_base64_str = "data:image/png;base64," + base64_str

    read_img = read_base64_image(full_base64_str)

    assert read_img.size == (20, 20)
    img_array = np.array(read_img)
    assert np.all(img_array == 127)
