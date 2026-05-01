import pytest
import sys
from unittest.mock import MagicMock, patch

@patch.dict(sys.modules, {
    'gradio': MagicMock(),
    'gradio_magicquill': MagicMock(),
    'torch': MagicMock(),
    'numpy': MagicMock(),
    'PIL': MagicMock(),
    'fastapi': MagicMock(),
    'uvicorn': MagicMock(),
    'MagicQuill': MagicMock(),
    'MagicQuill.llava_new': MagicMock(),
    'MagicQuill.scribble_color_edit': MagicMock(),
    'MagicQuill.folder_paths': MagicMock(),
})
def test_read_base64_image_unsupported_format():
    import gradio_run

    with pytest.raises(ValueError, match="Unsupported image format."):
        gradio_run.read_base64_image("data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

    with pytest.raises(ValueError, match="Unsupported image format."):
        gradio_run.read_base64_image("invalid_string")
