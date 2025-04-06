#!/bin/bash
echo "===== AI-Powered Job Application Screening System ====="

# Check if Ollama is running
echo "Checking if Ollama is running..."
if curl -s "http://localhost:11434/api/tags" > /dev/null; then
    echo "Ollama is running."
else
    echo "Ollama is not running. Starting Ollama..."
    # Try to start Ollama in the background
    if command -v ollama &> /dev/null; then
        ollama serve > /dev/null 2>&1 &
        echo "Waiting for Ollama to initialize (30 seconds)..."
        sleep 30
    else
        echo "Error: Ollama command not found. Please install Ollama or start it manually."
        echo "Visit https://ollama.ai/ for installation instructions."
        exit 1
    fi
fi

# Set the model to use
MODEL_TO_CHECK="gemma3:4b"
echo "Will use Gemma3:4b model."

# Check if the required model is available
echo "Checking if the required model $MODEL_TO_CHECK is available..."
if curl -s "http://localhost:11434/api/tags" | grep -q "\"name\":\"$MODEL_TO_CHECK\""; then
    echo "Model $MODEL_TO_CHECK is available."
else
    echo "Model $MODEL_TO_CHECK is not available. Pulling model..."
    curl -s -X POST "http://localhost:11434/api/pull" -d "{\"name\":\"$MODEL_TO_CHECK\"}" -H "Content-Type: application/json"
    # Wait for confirmation
    echo "Model download initiated. This may take some time depending on your internet connection."
    echo "Once downloaded, the model will be ready for use."
fi

echo "Starting the job screening pipeline..."
echo ""

# Run the optimized app
python optimized_app.py

echo ""
echo "Pipeline completed!"
echo "Press Enter to exit..."
read 