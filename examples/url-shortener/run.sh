#!/bin/bash

# URL Shortener - Startup Script

echo "🚀 Starting URL Shortener..."

# Install dependencies if not already installed
echo "📦 Checking dependencies..."
pip install -r requirements.txt -q

# Run the application
echo "🌐 Server starting at http://localhost:8000"
python main.py
