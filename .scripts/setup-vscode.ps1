# Setup VS Code settings for automatic venv activation
$projectRoot = Split-Path $PSScriptRoot -Parent
$vscodeDir = Join-Path $projectRoot ".vscode"
$settingsPath = Join-Path $vscodeDir "settings.json"

# Create .vscode directory if it doesn't exist
if (-not (Test-Path $vscodeDir)) {
    New-Item -ItemType Directory -Path $vscodeDir | Out-Null
    Write-Host "Created .vscode directory" -ForegroundColor Cyan
}

# Define the settings
$settings = @{
    "python.defaultInterpreterPath" = "`${workspaceFolder}/venv/Scripts/python.exe"
    "python.terminal.activateEnvironment" = $true
    "terminal.integrated.defaultProfile.windows" = "PowerShell (venv)"
    "terminal.integrated.env.windows" = @{
        "VIRTUAL_ENV" = "`${workspaceFolder}/venv"
    }
    "terminal.integrated.profiles.windows" = @{
        "PowerShell (venv)" = @{
            "source" = "PowerShell"
            "args" = @("-NoExit", "-Command", "& '`${workspaceFolder}/venv/Scripts/Activate.ps1'")
        }
    }
    "powershell.powerShellDefaultVersion" = "PowerShell (x64)"
    "powershell.integratedConsole.showOnStartup" = $false
}

# Convert to JSON and save
$settingsJson = $settings | ConvertTo-Json -Depth 3
$settingsJson | Set-Content -Path $settingsPath -Encoding UTF8

Write-Host "VS Code settings created at: $settingsPath" -ForegroundColor Green
Write-Host "Settings configured:" -ForegroundColor Cyan
Write-Host "  - Python interpreter: venv/Scripts/python.exe" -ForegroundColor White
Write-Host "  - Auto-activate environment: enabled" -ForegroundColor White
Write-Host "  - Default terminal: PowerShell" -ForegroundColor White
Write-Host ""
Write-Host "Restart VS Code or open a new terminal for changes to take effect." -ForegroundColor Yellow
