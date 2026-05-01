import torch
import numpy as np
import base64
import io
import sys
from PIL import Image
from unittest.mock import MagicMock

# Mock out heavy and unavailable dependencies globally before importing gradio_run.
# We use a context manager in the test functions to apply these mocks so they don't leak,
# but `gradio_run` imports these modules at the top level. Let's just create fake modules.

class FakeModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

sys.modules['gradio'] = FakeModule()
sys.modules['fastapi'] = FakeModule()
sys.modules['uvicorn'] = FakeModule()
sys.modules['gradio_magicquill'] = FakeModule()
sys.modules['MagicQuill.llava_new'] = FakeModule()
sys.modules['MagicQuill.scribble_color_edit'] = FakeModule()
sys.modules['MagicQuill.folder_paths'] = FakeModule()

from gradio_run import create_alpha_mask

def test_create_alpha_mask_rgb():
    # Create an RGB image (no alpha channel)
    img = Image.new('RGB', (10, 10), color=(255, 0, 0))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Test create_alpha_mask
    mask = create_alpha_mask(img_str)

    # Assert mask is correct shape, type and all zeros
    assert mask.shape == (1, 10, 10)
    assert mask.dtype == torch.float32
    assert torch.all(mask == 0.0)

def test_create_alpha_mask_rgba():
    # Create an RGBA image with varying alpha
    img = Image.new('RGBA', (10, 10), color=(255, 0, 0, 255))

    # Set a specific pixel's alpha to 127 (approx 0.5)
    img_data = np.array(img)
    img_data[5, 5, 3] = 127
    img_data[2, 2, 3] = 0
    img = Image.fromarray(img_data, 'RGBA')

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Test create_alpha_mask
    mask = create_alpha_mask(img_str)

    # Assert mask is correct shape and type
    assert mask.shape == (1, 10, 10)
    assert mask.dtype == torch.float32

    # Assert mask values are 1.0 - alpha
    # Alpha = 255 -> 1.0 -> mask = 0.0
    assert mask[0, 0, 0].item() == 0.0

    # Alpha = 127 -> 127/255 -> mask = 1.0 - 127/255 -> approx 0.50196
    assert abs(mask[0, 5, 5].item() - (1.0 - 127/255.0)) < 1e-5

    # Alpha = 0 -> 0.0 -> mask = 1.0
    assert mask[0, 2, 2].item() == 1.0
