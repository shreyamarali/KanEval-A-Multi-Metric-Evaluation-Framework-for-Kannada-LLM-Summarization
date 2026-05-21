#!/bin/bash

# KanEval Setup Script
# Sets up the environment and runs the Streamlit app

echo "================================================"
echo "   KanEval - Kannada LLM Evaluation Framework  "
echo "================================================"

# Check Python version
echo ""
echo "[1/4] Checking Python version..."
python_version=$(python3 --version 2>&1)
if [ $? -ne 0 ]; then
    echo "ERROR: Python3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi
echo "Found: $python_version"

# Create virtual environment
echo ""
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists, skipping."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "[3/4] Installing dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Dependencies installed successfully."

# Run the app
echo ""
echo "[4/4] Launching KanEval Streamlit app..."
echo "Visit http://localhost:8501 in your browser."
echo ""
streamlit run app.py
