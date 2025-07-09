# Document Processing Pipeline - Project Summary

## 🚀 Overview

The Document Processing Pipeline is a Streamlit application that leverages AI vision models and OCR to automate the extraction, structuring, and analysis of information from PDF documents. It is designed for insurance, forms, and other structured documents, providing a user-friendly interface for intelligent document understanding.

---

## 📁 Project Structure

```
document-processing-pipeline/
├── myapp.py            # Main Streamlit application
├── README.md           # Documentation and usage instructions
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (Groq API key)
├── venv/               # Conda or virtual environment
```

---

## ✅ Key Features

1. **PDF Upload & Conversion**

   - Upload PDF files via the Streamlit sidebar.
   - Converts the first page of the PDF to a high-resolution image for processing.

2. **Image Preprocessing & OCR**

   - Applies adaptive thresholding for optimal OCR results.
   - Uses Tesseract OCR and Groq's vision model for robust text extraction.
   - Extracts line-level text with bounding boxes and confidence scores.

3. **AI Vision Analysis**

   - Utilizes Groq's `meta-llama/llama-4-scout-17b-16e-instruct` model for multimodal document understanding.
   - Detects headers, sections, and document structure using recursive LLM calls.

4. **Header Detection & Segmentation**

   - Automatically identifies main headers and subheaders.
   - Segments document content into a nested, human-readable structure.
   - Uses LLM to clean and prettify each segment for display.

5. **Metadata Extraction**

   - Extracts user-defined metadata fields (e.g., policy number, dates, holder name) using LLM.
   - Outputs metadata as structured JSON.

6. **Interactive Q&A**

   - Users can ask natural language questions about the document.
   - The app finds and highlights the exact answer phrase in the document image.
   - Uses fuzzy string matching and visual highlighting for clarity.

7. **Advanced UI/UX**
   - Clean, responsive Streamlit interface with progress indicators and error handling.
   - Expandable sections for images, preprocessed views, and results.
   - Visual document preview and statistics.

---

## 🛠️ How to Run

1. **Create environment and install dependencies:**

   ```bash
   conda create -p venv python=3.10 -y
   conda activate ./venv
   pip install -r requirements.txt
   ```

2. **Start the application:**

   ```bash
   streamlit run my_app.py
   ```

3. **Access in browser:**  
   Go to [http://localhost:8501](http://localhost:8501)

4. **Usage Steps:**
   - Enter your Groq API key in the sidebar (or set in `.env`)
   - Upload a PDF document
   - Optionally configure metadata fields
   - Click buttons to detect headers, extract metadata, or ask questions

---

## 🔑 Configuration

- **Groq API Key**: Required for AI vision and text processing (set in `.env` or sidebar)
- **Metadata Fields**: Customizable in the sidebar (defaults provided)

---

## 📊 Application Flow

1. **Upload PDF** → 2. **Convert to Image** → 3. **Preprocess for OCR**
2. **AI Vision Analysis** → 5. **Header Detection & Segmentation**
3. **Metadata Extraction** → 7. **OCR Analysis** → 8. **Interactive Q&A**

---

## ✨ Summary

This pipeline transforms complex, unstructured documents into structured, searchable, and interactive data. It combines the power of modern LLMs, OCR, and a user-friendly Streamlit interface to automate document understanding and information extraction, making it ideal for insurance, finance, and enterprise document workflows.
