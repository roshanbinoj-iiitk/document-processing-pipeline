import os
import json
import io
import base64
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
from groq import Groq
import re
import streamlit as st
from datetime import datetime

# Streamlit page configuration
st.set_page_config(
    page_title="Document Processing Pipeline",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.error-message {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
def pdf_to_image(pdf_file, dpi=300):
    """Convert PDF to image"""
    try:
        # Save uploaded file temporarily
        with open("temp_pdf.pdf", "wb") as f:
            f.write(pdf_file.read())
        
        images = convert_from_path("temp_pdf.pdf", dpi=dpi, first_page=1, last_page=1)
        
        # Clean up temp file
        os.remove("temp_pdf.pdf")
        
        return images[0] if images else None
    except Exception as e:
        st.error(f"Error converting PDF to image: {e}")
        return None

def preprocess_image(pil_image):
    """Preprocess image for better OCR"""
    img_np = np.array(pil_image.convert('L'))
    thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(thresh)

def encode_image_to_base64(image):
    """Encode image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def call_vision_model_for_description(base64_image, groq_api_key, prompt="Print the exact content of this image without description."):
    """Call Groq vision model for image description"""
    if not groq_api_key:
        st.error("Error: Groq API key not found.")
        return None

    client = Groq(api_key=groq_api_key)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ],
        }
    ]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling vision model: {e}")
        return None

def get_line_level_boxes(pil_image):
    """Get line-level bounding boxes from OCR"""
    img = np.array(pil_image)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    lines = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 50 and data['text'][i].strip():
            key = (data['page_num'][i], data['block_num'][i], data['par_num'][i], data['line_num'][i])
            if key not in lines:
                lines[key] = []
            lines[key].append(i)

    line_boxes = []
    for indices in lines.values():
        x = min([data['left'][i] for i in indices])
        y = min([data['top'][i] for i in indices])
        w = max([data['left'][i] + data['width'][i] for i in indices]) - x
        h = max([data['top'][i] + data['height'][i] for i in indices]) - y
        text = " ".join([data['text'][i] for i in indices])
        line_boxes.append({'text': text.strip(), 'box': (x, y, w, h)})
    return line_boxes

# --- LLM Utilities ---
def ask_llm(prompt, groq_api_key, max_tokens=512):
    """Ask LLM a question"""
    if not groq_api_key:
        st.error("Error: Groq API key not found.")
        return None

    client = Groq(api_key=groq_api_key)
    messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

# --- Header Detection and Segmentation ---
def detect_headers(full_text, groq_api_key):
    """Detect headers in document"""
    prompt = f"""Identify the main headers and subheaders in the following text. Return them as a nested list or dictionary indicating the hierarchy.\nText:\n{full_text}"""
    return ask_llm(prompt, groq_api_key, max_tokens=1024)

def segment_document(full_text, headers_structure):
    """Segment document by headers"""
    segments = {}
    if headers_structure:
        current_header = "Introduction"
        segments[current_header] = []
        lines = full_text.split('\n')
        for line in lines:
            is_header = False
            # Simplified header detection logic
            if line.strip() and (line.isupper() or line.strip().endswith(':')):
                current_header = line.strip()
                segments.setdefault(current_header, [])
                is_header = True
            if not is_header and line.strip():
                segments.setdefault(current_header, []).append(line.strip())
    else:
        segments["Full Document"] = full_text.split('\n')
    return segments

# --- Metadata Extraction ---
def extract_metadata(full_text, metadata_fields, groq_api_key):
    """Extract metadata from document"""
    prompt = f"""Extract the following metadata from the text: {', '.join(metadata_fields)}.
Text:
{full_text}
Return the metadata as a JSON object where the keys are snake_case versions of the fields and values are strings or null."""

    llm_response = ask_llm(prompt, groq_api_key, max_tokens=512)

    if not llm_response:
        return {field: None for field in metadata_fields}

    try:
        # Extract the first JSON-like structure in the response
        json_match = re.search(r'\{[\s\S]*?\}', llm_response)
        if not json_match:
            raise ValueError("No JSON object found in response.")

        cleaned_json = json_match.group(0)
        parsed = json.loads(cleaned_json)

        key_map = {field: field.lower().replace(' ', '_') for field in metadata_fields}
        return {field: parsed.get(key_map[field], None) for field in metadata_fields}

    except Exception as e:
        st.warning(f"Could not parse metadata response as JSON. Error: {e}")
        return {field: None for field in metadata_fields}

def highlight_best_match(pil_image, line_boxes, answer_text):
    """Highlight best matching text in image"""
    img = np.array(pil_image.convert("RGB"))
    matches_above_threshold = []

    for item in line_boxes:
        score = fuzz.token_set_ratio(answer_text.lower(), item['text'].lower())
        if score > 80:  # Increase threshold for stricter match
            matches_above_threshold.append(item)

    if matches_above_threshold:
        for item in matches_above_threshold:
            x, y, w, h = item['box']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        st.success("✅ Matches highlighted in green.")
    else:
        st.warning("⚠️ No strong match found.")

    return Image.fromarray(img)

def ask_question_about_document(text, pil_image, line_boxes, question, groq_api_key):
    """Ask a question about the document and highlight the answer"""
    prompt = f"""
You are given OCR text extracted from an insurance document.

OCR Text:
{text}

Now answer this question: "{question}"

Return only the **exact phrase** from the OCR text that answers the question. Do not paraphrase or explain.
"""
    answer = ask_llm(prompt, groq_api_key, max_tokens=256)
    
    if answer:
        highlighted_image = highlight_best_match(pil_image, line_boxes, answer)
        return answer, highlighted_image
    
    return None, pil_image

# --- MAIN STREAMLIT APP ---
def main():
    st.markdown('<h1 class="main-header">📄 Document Processing Pipeline</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.title("🔧 Configuration")
    
    # API Key input
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key to use the vision and text models"
    )
    
    # File upload
    st.sidebar.subheader("📤 Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document for processing"
    )
    
    # Metadata fields configuration
    st.sidebar.subheader("🏷️ Metadata Fields")
    default_fields = ["policy number", "policy start date", "policy end date", "policy holder name"]
    metadata_fields_input = st.sidebar.text_area(
        "Metadata fields to extract (one per line)",
        value="\n".join(default_fields),
        help="Enter the metadata fields you want to extract, one per line"
    )
    metadata_fields = [field.strip() for field in metadata_fields_input.split('\n') if field.strip()]
    
    # Main content
    if not uploaded_file:
        st.info("👆 Please upload a PDF file using the sidebar to start processing.")
        
        # Demo section
        st.subheader("🎯 What This App Does")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📄 Document Support**
            - PDF files
            - Image extraction
            - OCR text extraction
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Features**
            - Vision model analysis
            - Header detection
            - Metadata extraction
            """)
        
        with col3:
            st.markdown("""
            **💬 Interactive Q&A**
            - Ask questions about document
            - Visual answer highlighting
            - Exact text matching
            """)
        
        return
    
    if not groq_api_key:
        st.error("🔑 Please enter your Groq API key in the sidebar to proceed.")
        return
    
    # Process uploaded file
    st.subheader(f"📋 Processing: {uploaded_file.name}")
    
    with st.spinner("Converting PDF to image..."):
        pil_image = pdf_to_image(uploaded_file)
    
    if pil_image is None:
        st.error("❌ Failed to convert PDF to image.")
        return
    
    # Display original image
    st.subheader("📸 Original Document")
    st.image(pil_image, caption="Original PDF Page", use_container_width=True)
    
    # Preprocess image
    with st.spinner("Preprocessing image for OCR..."):
        preprocessed_image = preprocess_image(pil_image)
    
    # Show preprocessed image
    with st.expander("🔧 Preprocessed Image (for OCR)"):
        st.image(preprocessed_image, caption="Preprocessed Image", use_container_width=True)
    
    # Get vision model description
    with st.spinner("Getting document description from vision model..."):
        base64_image = encode_image_to_base64(preprocessed_image)
        description = call_vision_model_for_description(base64_image, groq_api_key)
    
    if not description:
        st.error("❌ Could not get description from vision model.")
        return
    
    # Display vision model output
    st.subheader("🤖 Vision Model Description")
    st.text_area("Extracted Text", description, height=300)
    
    # Header detection
    with st.spinner("Detecting document headers..."):
        headers_structure = detect_headers(description, groq_api_key)
    
    if headers_structure:
        st.subheader("📋 Header Structure")
        st.text_area("Detected Headers", headers_structure, height=150)
        
        # Document segmentation
        document_segments = segment_document(description, headers_structure)
        
        st.subheader("📑 Document Segments")
        for header, content in document_segments.items():
            with st.expander(f"Section: {header}"):
                st.write("\n".join(content[:10]))  # Show first 10 lines
    
    # Metadata extraction
    with st.spinner("Extracting metadata..."):
        metadata = extract_metadata(description, metadata_fields, groq_api_key)
    
    st.subheader("🏷️ Extracted Metadata")
    metadata_df = [{"Field": field, "Value": value} for field, value in metadata.items()]
    st.table(metadata_df)
    
    # Get line boxes for highlighting
    with st.spinner("Analyzing document structure..."):
        line_boxes = get_line_level_boxes(preprocessed_image)
    
    # Interactive Q&A section
    st.subheader("💬 Ask Questions About the Document")
    
    question = st.text_input(
        "What would you like to know about this document?",
        placeholder="e.g., What is the policy number?"
    )
    
    if question:
        with st.spinner(f"Finding answer to: {question}"):
            answer, highlighted_image = ask_question_about_document(
                description, pil_image, line_boxes, question, groq_api_key
            )
        
        if answer:
            st.subheader("📝 Answer")
            st.success(f"**Answer:** {answer}")
            
            st.subheader("🎯 Highlighted Document")
            st.image(highlighted_image, caption=f"Answer highlighted for: {question}", use_container_width=True)
        else:
            st.error("❌ Could not find an answer to your question.")
    
    # Additional features
    with st.expander("🔍 Advanced Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Document Statistics")
            word_count = len(description.split())
            char_count = len(description)
            line_count = len(description.split('\n'))
            
            st.metric("Word Count", word_count)
            st.metric("Character Count", char_count)
            st.metric("Line Count", line_count)
        
        with col2:
            st.subheader("🔤 OCR Confidence")
            st.info(f"Found {len(line_boxes)} text regions with high confidence")
            
            if line_boxes:
                avg_length = sum(len(box['text']) for box in line_boxes) / len(line_boxes)
                st.metric("Average Text Length per Region", f"{avg_length:.1f} chars")

if __name__ == "__main__":
    main()

