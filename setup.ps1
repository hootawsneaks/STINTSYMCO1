Write-Host "May God be with ye."
$filename = ".venv"
$targetDir = Join-Path -Path "." -ChildPath $filename

# if .venv exists
if (Test-Path -Path $targetDir) {
	.\.venv\Scripts\Activate.ps1
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
	uv pip install ultralytics
   uv pip install jupyter
	Write-Host "Enter the venv with: .venv\Scripts\activate"
} 
else {
	if (-not (Get-Command uv -ErrorAction SilentlyContinue) ) {
		powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
		# Refresh Path
		$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
	}
	uv venv
	.\.venv\Scripts\Activate.ps1
	uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
	uv pip install ultralytics
   uv pip install jupyter
        Write-Host "Enter the venv with: .venv\Scripts\activate"
}

  