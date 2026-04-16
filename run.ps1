# PowerShell script for YOLOv8 Live Object Detection and Tracking
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host " YOLOv8 Live Object Detection and Tracking" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Setting up environment..." -ForegroundColor Yellow
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "Environment variable KMP_DUPLICATE_LIB_OK set to TRUE" -ForegroundColor Green
Write-Host ""
Write-Host "Starting the application..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Make sure you have:" -ForegroundColor White
Write-Host "1. Installed Python 3.8+" -ForegroundColor White
Write-Host "2. Installed dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "3. Connected a webcam to your computer" -ForegroundColor White
Write-Host ""
Write-Host "The application will start on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

python app.py

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
