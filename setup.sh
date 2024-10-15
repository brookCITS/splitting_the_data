#!/bin/bash
echo "Setting up Python project..."
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo "Setup complete. Run 'pytest' to test your code."
