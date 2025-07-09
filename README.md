# Document Processing Pipeline

A Streamlit application for intelligent document processing using AI vision models and OCR.

---

## 🚀 Features

- **PDF Document Processing**: Upload and process PDF files
- **AI Vision Analysis**: Uses Groq's vision models for document understanding
- **OCR Text Extraction**: Extracts text with confidence scoring and bounding boxes
- **Header Detection**: Automatically identifies document structure and segments content
- **Metadata Extraction**: Extracts custom metadata fields using AI
- **Interactive Q&A**: Ask questions about the document with visual answer highlighting

---

## 🛠️ Installation

### Prerequisites

- Python 3.10
- [conda](https://docs.conda.io/en/latest/) (recommended)
- Tesseract OCR
- poppler-utils (for PDF processing)

### System Dependencies

#### Ubuntu/Debian

```sh
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

#### Arch Linux/EndeavourOS

```sh
sudo pacman -S tesseract poppler
```

#### macOS

```sh
brew install tesseract poppler
```

### Python Environment

1. **Create a conda environment:**

   ```sh
   conda create -p venv python=3.10 -y
   conda activate ./venv
   ```

2. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## ⚙️ Configuration

### Groq API Key

- Sign up at [Groq Console](https://console.groq.com/) and create an API key.
- Add your API key to a `.env` file as:
  ```
  GROQ_API_KEY="your_groq_api_key_here"
  ```
- The app will use this key automatically, or you can enter it in the sidebar.

---

## 🚦 Usage

1. Start the Streamlit application:

   ```sh
   streamlit run my_app.py
   ```

2. Open your browser and go to [http://localhost:8501](http://localhost:8501)

3. Enter your Groq API key in the sidebar (if not loaded from `.env`)

4. Upload a PDF document

5. Configure metadata fields to extract (optional)

6. Process the document and explore results

7. Ask questions about the document in the Q&A section

---

## 📝 Application Workflow

1. **PDF to Image Conversion**: Converts the first page of the PDF to a high-resolution image
2. **Image Preprocessing**: Applies adaptive thresholding for better OCR results
3. **Vision Model Analysis**: Uses Groq's vision model to extract and understand document content
4. **Header Detection**: Identifies document structure and sections
5. **Metadata Extraction**: Extracts specified metadata fields using AI
6. **OCR Analysis**: Performs line-level OCR with confidence scoring
7. **Interactive Q&A**: Allows users to ask questions with visual answer highlighting

---

## 📄 Supported Document Types

- PDF files (first page processed)
- Insurance documents
- Forms and structured documents
- Text-heavy documents

---

## 🏷️ Default Metadata Fields

- Policy Number
- Policy Start Date
- Policy End Date
- Policy Holder Name

(You can customize these in the sidebar)

---

## ⚡ Technical Details

- **Main Application File**: `my_app.py`
- **Vision Model**: `meta-llama/llama-4-scout-17b-16e-instruct` (via Groq API)
- **OCR Engine**: Tesseract with adaptive thresholding
- **Text Matching**: Fuzzy string matching (fuzzywuzzy)
- **UI Framework**: Streamlit

---

## 🛠️ Troubleshooting

- **Tesseract not found**: Install `tesseract-ocr` system package
- **PDF conversion fails**: Install `poppler-utils` system package
- **API errors**: Check your Groq API key and internet connection
- **Poor OCR results**: Try uploading a higher quality PDF

---

## 📜 License

MIT License

---

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request or open
