#!/bin/bash

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Seed the database
echo "Seeding database with sample data..."
python seed.py

# Start the server
echo "Starting server on http://localhost:8000"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
