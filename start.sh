#!/bin/bash

echo "=========================================="
echo "   DeepGuard AI - Starting Server"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Navigate to script directory
cd "$(dirname "$0")"

# Navigate to backend directory
cd backend

# Check if virtual environment exists, if not create one
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r ../requirements.txt -q

# Create uploads directory
mkdir -p uploads

# Start the server
echo ""
echo "Starting DeepGuard AI Server..."
echo "Dashboard will be available at: http://localhost:5000"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

python api.py

# Deactivate virtual environment on exit
deactivate