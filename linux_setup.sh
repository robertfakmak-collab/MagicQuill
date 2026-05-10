#!/bin/bash

CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "MagicQuill"; then
    echo "Creating conda environment MagicQuill..."
    conda create -n MagicQuill python=3.10 -y

    # Initialize conda after creating new environment
    echo "Initializing conda..."
    conda init bash

    echo "Environment created. Please run this script again to continue installation."
    exit 0
fi

# Activate conda environment
echo "Activating conda environment..."
conda activate MagicQuill

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch 2.1.2 with CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation and CUDA availability
python -c "import torch; print('PyTorch version:', torch.version); print('CUDA available:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "Failed to install or verify PyTorch"
    exit 1
fi

# Install the required interface package
echo "Installing gradio_magicquill..."
pip install gradio_magicquill-0.0.1-py3-none-any.whl

# Install the llava environment
echo "Setting up LLaVA environment..."
if [ ! -d "MagicQuill/LLaVA" ]; then
    echo "Directory MagicQuill/LLaVA does not exist. Ensure the folder structure is correct."
    exit 1
fi
cp -f pyproject.toml MagicQuill/LLaVA/
pip install -e MagicQuill/LLaVA/

# Install remaining dependencies
echo "Installing additional requirements..."
pip install -r requirements.txt

# Create desktop shortcut
echo "Creating desktop shortcut..."
if command -v xdg-user-dir &> /dev/null; then
    DESKTOP_DIR=$(xdg-user-dir DESKTOP)
else
    DESKTOP_DIR="$HOME/Desktop"
fi

SHORTCUT_PATH="$DESKTOP_DIR/MagicQuill.desktop"

# Create Desktop directory if it doesn't exist
mkdir -p "$DESKTOP_DIR"

PYTHON_EXEC=$(which python)
REPO_DIR=$(pwd)

cat > "$SHORTCUT_PATH" << EOL
[Desktop Entry]
Version=1.0
Type=Application
Name=MagicQuill
Comment=Launch MagicQuill Interface
Exec=bash -c 'cd "$REPO_DIR" && export CUDA_VISIBLE_DEVICES=0 && "$PYTHON_EXEC" gradio_run.py'
Icon=utilities-terminal
Terminal=true
Categories=Graphics;
EOL

chmod +x "$SHORTCUT_PATH"
echo "Desktop shortcut created at $SHORTCUT_PATH"

# Run MagicQuill
echo "Starting MagicQuill..."
export CUDA_VISIBLE_DEVICES=0
python gradio_run.py || {
    echo "Error: Failed to run MagicQuill."
    exit 1
}
