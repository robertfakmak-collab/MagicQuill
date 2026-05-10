# Create conda environment if it doesn't exist
$condaEnvList = conda env list | Out-String
if ($condaEnvList -notmatch "MagicQuill") {
    Write-Host "Creating conda environment MagicQuill..."
    conda create -n MagicQuill python=3.10 -y
    # Initialize conda
    Write-Host "Initializing conda..."
    conda init powershell
    Write-Host "Environment created. Please restart the script."
    Read-Host -Prompt "Press Enter to continue"
    exit 0
}

# Activate conda environment
Write-Host "Activating conda environment..."
& conda shell.powershell activate MagicQuill | Invoke-Expression

# Install PyTorch with CUDA 11.8
Write-Host "Installing PyTorch 2.1.2 with CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch installation and CUDA availability
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install or verify PyTorch"
    Read-Host -Prompt "Press Enter to continue"
    exit 1
}

# Install the required interface package
Write-Host "Installing gradio_magicquill..."
pip install gradio_magicquill-0.0.1-py3-none-any.whl

# Install the llava environment
Write-Host "Setting up LLaVA environment..."
if (-not (Test-Path "LLaVA")) {
    Write-Host "Directory LLaVA does not exist. Ensure the folder structure is correct."
    exit 1
}
Copy-Item -Path pyproject.toml -Destination "LLaVA" -Force
pip install -e LLaVA\

# Install remaining dependencies
Write-Host "Installing additional requirements..."
pip install -r requirements.txt

# Run MagicQuill
Write-Host "Starting MagicQuill..."
$env:CUDA_VISIBLE_DEVICES="0"
python gradio_run.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to run MagicQuill."
    Read-Host -Prompt "Press Enter to continue"
    exit 1
}
