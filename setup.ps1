Write-Host "May God be with ye."
$filename = ".venv"
$targetDir = Join-Path -Path "." -ChildPath $filename

# if .venv exists
if (Test-Path -Path $targetDir) {
	.\.venv\Scripts\Activate.ps1
	# Use cuda if ya'll have it.
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    uv pip install ultralytics jupyter albumentations scikit-learn tqdm matplotlib opencv-python
	Write-Host "Enter the venv with: .venv\Scripts\activate"
	$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
} 
else {
	if (-not (Get-Command uv -ErrorAction SilentlyContinue) ) {
		powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
		# Refresh Path
		$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
	}
	uv venv
	.\.venv\Scripts\Activate.ps1
	# Use Cuda tho if ya'll got it
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
	uv pip install ultralytics jupyter albumentations scikit-learn tqdm matplotlib opencv-python
        Write-Host "Enter the venv with: .venv\Scripts\activate"
}

  