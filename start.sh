#!/bin/bash

# --- AI Skill Agent Startup Script ---

set -e

echo "🚀 Starting AI Skill Agent Setup..."

# 1. Environment Check
if [ ! -f .env ]; then
    echo "📄 .env file not found. Creating from .env.example..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Please update the GOOGLE_API_KEY in your .env file."
fi

# 2. Docker Execution
echo "🐳 Building and starting Docker containers..."
docker compose up --build -d

echo "------------------------------------------------"
echo "✅ AI Skill Agent is running!"
echo "------------------------------------------------"
echo "🖥️  Frontend (UI): http://localhost:8501"
echo "🔌 Backend (API):  http://localhost:8000"
echo "📚 API Docs:      http://localhost:8000/docs"
echo "------------------------------------------------"
echo "💡 Tip: Use 'docker compose logs -f' to see real-time logs."
echo "💡 Tip: Use 'docker compose down' to stop the application."
