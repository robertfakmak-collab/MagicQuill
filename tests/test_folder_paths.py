import pytest
import MagicQuill.folder_paths as folder_paths

@pytest.fixture
def restore_folder_paths():
    # Save original states
    orig_output = folder_paths.get_output_directory()
    orig_temp = folder_paths.get_temp_directory()
    orig_input = folder_paths.get_input_directory()

    yield

    # Restore original states
    folder_paths.set_output_directory(orig_output)
    folder_paths.set_temp_directory(orig_temp)
    folder_paths.set_input_directory(orig_input)

def test_set_get_output_directory(restore_folder_paths):
    test_dir = "/path/to/output"
    folder_paths.set_output_directory(test_dir)
    assert folder_paths.get_output_directory() == test_dir

def test_set_get_temp_directory(restore_folder_paths):
    test_dir = "/path/to/temp"
    folder_paths.set_temp_directory(test_dir)
    assert folder_paths.get_temp_directory() == test_dir

def test_set_get_input_directory(restore_folder_paths):
    test_dir = "/path/to/input"
    folder_paths.set_input_directory(test_dir)
    assert folder_paths.get_input_directory() == test_dir
