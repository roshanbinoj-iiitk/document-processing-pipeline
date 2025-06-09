# Document Processing Pipeline

A Streamlit application for intelligent document processing using AI vision models and OCR.

## Features

- 📄 **PDF Document Processing**: Upload and process PDF documents
- 🤖 **AI Vision Analysis**: Uses Groq's vision models to understand document content
- 🔍 **OCR Text Extraction**: Extracts text with confidence scoring
- 📋 **Header Detection**: Automatically identifies document structure
- 🏷️ **Metadata Extraction**: Extracts custom metadata fields
- 💬 **Interactive Q&A**: Ask questions about the document with visual highlighting
- 🎯 **Visual Answer Highlighting**: Highlights relevant text regions in the document

## Installation

### Prerequisites

- Python 3.8+
- uv package manager
- Tesseract OCR
- poppler-utils (for PDF processing)

### System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

#### Arch Linux/EndeavourOS:
```bash
sudo pacman -S tesseract poppler
```

#### macOS:
```bash
brew install tesseract poppler
```

### Python Dependencies

1. Clone or create the project directory:
```bash
mkdir document-processing-app
cd document-processing-app
```

2. Initialize uv project:
```bash
uv init
```

3. Install dependencies:
```bash
uv add streamlit opencv-python numpy pytesseract matplotlib pillow pdf2image fuzzywuzzy groq python-levenshtein
```

## Configuration

### Groq API Key

You'll need a Groq API key to use the vision and text models:

1. Sign up at [Groq Console](https://console.groq.com/)
2. Create an API key
3. Enter the API key in the Streamlit sidebar when running the app

## Usage

1. Start the Streamlit application:
```bash
uv run streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter your Groq API key in the sidebar

4. Upload a PDF document

5. Configure metadata fields to extract (optional)

6. Process the document and explore the results

7. Ask questions about the document in the Q&A section

## Application Workflow

1. **PDF to Image Conversion**: Converts the first page of the PDF to a high-resolution image
2. **Image Preprocessing**: Applies adaptive thresholding for better OCR results
3. **Vision Model Analysis**: Uses Groq's vision model to extract and understand document content
4. **Header Detection**: Identifies document structure and sections
5. **Metadata Extraction**: Extracts specified metadata fields using AI
6. **OCR Analysis**: Performs line-level OCR with confidence scoring
7. **Interactive Q&A**: Allows users to ask questions with visual answer highlighting

## Supported Document Types

- PDF files (first page processed)
- Insurance documents
- Forms and structured documents
- Text-heavy documents

## Default Metadata Fields

- Policy Number
- Policy Start Date
- Policy End Date
- Policy Holder Name

(These can be customized in the sidebar)

## Technical Details

### AI Models Used

- **Vision Model**: `meta-llama/llama-4-maverick-17b-128e-instruct` for document understanding
- **Text Model**: `meta-llama/llama-4-maverick-17b-128e-instruct` for text processing and Q&A

### OCR Configuration

- Uses Tesseract OCR with adaptive thresholding
- Confidence threshold: 50%
- Line-level text extraction with bounding boxes

### Text Matching

- Uses fuzzy string matching (fuzzywuzzy)
- Match threshold: 80% for answer highlighting
- Token set ratio for flexible matching

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Install tesseract-ocr system package
2. **PDF conversion fails**: Install poppler-utils system package
3. **API errors**: Check your Groq API key and internet connection
4. **Poor OCR results**: Try uploading a higher quality PDF

### Performance Tips

- Use high-resolution PDFs for better OCR results
- Ensure good contrast in source documents
- The app processes only the first page of multi-page PDFs

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

