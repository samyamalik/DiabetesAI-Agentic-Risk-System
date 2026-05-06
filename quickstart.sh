#!/bin/bash

# Quick Start Guide for Diabetes AI Agent System
# This script sets up the environment and runs the system

echo "🏥 Diabetes AI Agent System - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"
echo ""

# Create virtual environment
echo "✓ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate
echo "  Virtual environment activated"
echo ""

# Install dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "  Dependencies installed"
echo ""

# Run the main pipeline
echo "✓ Running main pipeline..."
python3 -c "
from src.main_pipeline import DiabetesAIAgentsPipeline
pipeline = DiabetesAIAgentsPipeline()
results = pipeline.run_complete_pipeline(train_model=True)
print('\\n✅ Pipeline execution complete!')
"
echo ""

# Launch Streamlit app (optional)
read -p "Do you want to launch the Streamlit web app? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Launching Streamlit app..."
    streamlit run app/streamlit_app.py
fi

echo ""
echo "🎉 Setup complete! Happy analyzing!"
