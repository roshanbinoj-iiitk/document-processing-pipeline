#!/bin/bash

# Document Processing Pipeline Startup Script

echo "🚀 Starting Document Processing Pipeline..."
echo "📋 Checking dependencies..."

# Check if tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "❌ Tesseract OCR not found. Please install it:"
    echo "   sudo pacman -S tesseract"  # For Arch/EndeavourOS
    exit 1
fi

# Check if pdftoppm is available (from poppler)
if ! command -v pdftoppm &> /dev/null; then
    echo "❌ Poppler utilities not found. Please install them:"
    echo "   sudo pacman -S poppler"  # For Arch/EndeavourOS
    exit 1
fi

echo "✅ All system dependencies found!"
echo "🌐 Starting Streamlit application..."
echo "📖 Open your browser and navigate to: http://localhost:8501"
echo "🔑 Don't forget to enter your Groq API key in the sidebar!"
echo ""

# Start the Streamlit app
uv run streamlit run app.py

