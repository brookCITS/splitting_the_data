#!/bin/bash
echo "Setting up Python project..."

# Check if a virtual environment already exists
if [ -d "venv" ]; then
  echo "Virtual environment 'venv' already exists."
else
  # Create a new virtual environment
  echo "Creating a new virtual environment..."
  python3 -m venv venv
  echo "Virtual environment 'venv' created."
fi

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
else
  echo "No requirements.txt file found. Please add your dependencies."
fi

echo "Setup is complete. Virtual environment is ready."