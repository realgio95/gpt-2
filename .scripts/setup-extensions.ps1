# setup powershell extension in vscode
$projectRoot = Split-Path $PSScriptRoot -Parent
$vscodeDir = Join-Path $projectRoot ".vscode"
$extensionsPath = Join-Path $vscodeDir "extensions.json"
# Create .vscode directory if it doesn't exist
if (-not (Test-Path $vscodeDir)) {
    New-Item -ItemType Directory -Path $vscodeDir | Out-Null
    Write-Host "Created .vscode directory" -ForegroundColor Cyan
}
# Define the extensions
$extensions = @{
    "recommendations" = @(
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-vscode.powershell"
    )
}
# Convert to JSON and save
$extensionsJson = $extensions | ConvertTo-Json -Depth 2
$extensionsJson | Set-Content -Path $extensionsPath -Encoding UTF8
Write-Host "VS Code extensions recommendations created at: $extensionsPath" -ForegroundColor Green
Write-Host "Recommended extensions:" -ForegroundColor Cyan
Write-Host "  - ms-python.python" -ForegroundColor White
Write-Host "  - ms-toolsai.jupyter" -ForegroundColor White
Write-Host "  - ms-vscode.powershell" -ForegroundColor White
Write-Host ""Write-Host "You can install these extensions in VS Code by clicking on the lightbulb icon

Restart VS Code or open a new terminal for changes to take effect." -ForegroundColor Yellow
Write-Host "You can install these extensions in VS Code by clicking on the lightbulb icon that appears when you open a Python or PowerShell file." -ForegroundColor Yellow

# Let this Ps1 actually install when run by:
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-vscode.powershell
