@echo off
echo ===== AI-Powered Job Application Screening System =====

echo Checking if Ollama is running...

REM Check if Ollama is running by making a request to its API
powershell -Command "try { $response = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5; if ($response.StatusCode -eq 200) { Write-Host 'Ollama is running.' } } catch { Write-Host 'Ollama is not running. Starting Ollama...'; Start-Process 'ollama.exe' -WindowStyle Minimized; Write-Host 'Waiting for Ollama to initialize (30 seconds)...'; Start-Sleep -s 30 }"

REM Set the model to use
set MODEL_TO_CHECK=gemma3:4b
echo Will use Gemma3:4b model.

echo Checking if the required model %MODEL_TO_CHECK% is available...
powershell -Command "$models = (Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 10).Content | ConvertFrom-Json; $modelExists = $false; foreach ($model in $models.models) { if ($model.name -eq '%MODEL_TO_CHECK%') { $modelExists = $true; break; } }; if (-not $modelExists) { Write-Host 'Model %MODEL_TO_CHECK% is not available. Pulling model...'; Invoke-WebRequest -Uri 'http://localhost:11434/api/pull' -Method Post -Body '{\"name\":\"%MODEL_TO_CHECK%\"}' -ContentType 'application/json'; } else { Write-Host 'Model %MODEL_TO_CHECK% is available.' }"

echo Starting the job screening pipeline...
echo.

REM Run the optimized app
python optimized_app.py

echo.
echo Pipeline completed!
pause 