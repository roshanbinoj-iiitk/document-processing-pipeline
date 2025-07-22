import os
import json
import io
import base64
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
from groq import Groq
import re
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

import re

def clean_llm_ocr_text(text: str) -> str:
    # Replace HTML <br> tags with newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    # Remove stray pipes and spaces at line ends
    text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)
    # Remove lines that are just pipes or dashes (table separators)
    text = re.sub(r'^\s*\|?\s*-+\s*\|?.*$', '', text, flags=re.MULTILINE)
    # Remove weird unicode or broken table chars
    text = re.sub(r'[∣′,′]', '', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    # Fix broken lines: join lines that are not headers or table rows
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # If line looks like a table row, keep as is
        if re.match(r'^\s*\|.*\|.*\|', line):
            cleaned_lines.append(line.strip())
        # If line is a header or section, keep as is
        elif re.match(r'^[A-Z][A-Za-z0-9\s\.\-/&]+:$', line.strip()):
            cleaned_lines.append(f"\n**{line.strip()}**\n")
        else:
            # Join short lines to previous if not empty
            if cleaned_lines and len(line.strip()) < 40 and line.strip() and not cleaned_lines[-1].endswith('.'):
                cleaned_lines[-1] += ' ' + line.strip()
            else:
                cleaned_lines.append(line.strip())
    # Remove empty lines at start/end
    cleaned = '\n'.join([l for l in cleaned_lines if l.strip()])
    # Optional: format tables as Markdown
    cleaned = re.sub(r'\| ([^|]+) \| ([^|]+) \|', r'**\1:** \2', cleaned)
    return cleaned

load_dotenv()

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
        with open("temp_pdf.pdf", "wb") as f:
            f.write(pdf_file.read())
        images = convert_from_path("temp_pdf.pdf", dpi=dpi, first_page=1, last_page=1)
        os.remove("temp_pdf.pdf")
        return images[0] if images else None
    except Exception as e:
        st.error(f"Error converting PDF to image: {e}")
        return None
    
def pdf_to_images(pdf_file, dpi=300):
    """Convert PDF to list of images (all pages)"""
    try:
        with open("temp_pdf.pdf", "wb") as f:
            f.write(pdf_file.read())
        images = convert_from_path("temp_pdf.pdf", dpi=dpi)
        os.remove("temp_pdf.pdf")
        return images
    except Exception as e:
        st.error(f"Error converting PDF to images: {e}")
        return []

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
    """Call Groq vision model for image description (multimodal OCR)"""
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
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling vision model: {e}")
        return None

def multimodal_object_detection(base64_image, groq_api_key):
    """Use multimodal LLM to detect text regions and bounding boxes. Always return a list of dicts with 'text' and 'box'."""
    prompt = (
        "Detect all text regions in this document image. "
        "For each region, return a JSON object with the text and its bounding box as [x, y, w, h]. "
        "Return a list of such objects. Only output JSON."
    )
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
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
        
        )
        response = completion.choices[0].message.content
        # Try to extract JSON list
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                detected = json.loads(json_match.group(0))
                # Normalize: ensure each item has 'text' and 'box'
                norm = []
                for item in detected:
                    if isinstance(item, dict) and 'text' in item and 'box' in item and isinstance(item['box'], list) and len(item['box']) == 4:
                        norm.append({'text': str(item['text']), 'box': [int(x) for x in item['box']]})
                if norm:
                    return norm
            except Exception as json_e:
                st.warning(f"LLM returned invalid JSON for object detection. Falling back to OCR. (Error: {json_e})")
                # Optionally, for debugging:
                # st.expander("LLM Raw Response").write(response)
        else:
            st.warning("LLM did not return a JSON list for object detection. Falling back to OCR.")
        return []
    except Exception as e:
        st.warning(f"Object detection failed: {e}. Falling back to OCR.")
        return []

def get_line_level_boxes(pil_image, base64_image, groq_api_key):
    """Get line-level bounding boxes from multimodal LLM object detection or fallback to Tesseract."""
    # Try LLM-based detection first
    detected = multimodal_object_detection(base64_image, groq_api_key)
    if detected and isinstance(detected, list) and all('text' in d and 'box' in d for d in detected):
        return detected
    # Fallback to Tesseract if LLM fails or returns nothing
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
        line_boxes.append({'text': text.strip(), 'box': [int(x), int(y), int(w), int(h)]})
    return line_boxes

# --- LLM Utilities ---
def ask_llm(prompt, groq_api_key):
    """Ask LLM a question"""
    if not groq_api_key:
        st.error("Error: Groq API key not found.")
        return None

    client = Groq(api_key=groq_api_key)
    messages = [{"role": "user", "content": prompt}]

    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling LLM: {e}")
        return None

# --- Header Detection and Segmentation ---
def detect_headers_recursive(full_text, groq_api_key, depth=0):
    """Recursively detect headers and build nested structure using LLM"""
    prompt = (
        "Analyze the following document text and identify the main headers and subheaders. "
        "Return a JSON object representing the hierarchy, where each header may have a 'subsections' key "
        "with a list of nested headers. Only output JSON.\n\n"
        f"Text:\n{full_text}"
    )
    response = ask_llm(prompt, groq_api_key)
    if not response:
        return {}
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON object found in response.")
        return json.loads(json_match.group(0))
    except Exception as e:
        if depth == 0:
            st.warning(f"Header detection failed: {e}")
        return {}

def segment_document_by_headers(full_text, header_structure):
    """
    Recursively organize text into chunks based on the nested header hierarchy.
    Returns a nested dict: {header: {"text": [...], "subsections": {...}}}
    """
    lines = full_text.split('\n')

    def assign_lines(headers, lines, start_idx=0):
        segments = {}
        idx = start_idx
        for header, content in headers.items():
            section = {"text": [], "subsections": {}}
            # If content has subsections, recurse
            if isinstance(content, dict) and "subsections" in content:
                # Find where this header starts in lines
                while idx < len(lines) and header.strip().lower() not in lines[idx].strip().lower():
                    idx += 1
                idx += 1  # skip header line
                # Recursively assign lines to subsections
                section["subsections"], idx = assign_lines(content["subsections"], lines, idx)
            else:
                # Collect lines until next header or end
                section_lines = []
                while idx < len(lines):
                    line = lines[idx].strip()
                    # Stop if line matches any header at this level
                    if any(h.lower() in line.lower() for h in headers if h != header):
                        break
                    section_lines.append(line)
                    idx += 1
                section["text"] = section_lines
            segments[header] = section
        return segments, idx

    segments, _ = assign_lines(header_structure, lines)
    return segments

def render_segments(segments, level=1):
    for header, content in segments.items():
        st.markdown(f"{'#' * min(level,6)} {header}")
        if content["text"]:
            st.markdown("<br>".join([line for line in content["text"] if line]), unsafe_allow_html=True)
        if content["subsections"]:
            render_segments(content["subsections"], level + 1)

# --- Metadata Extraction ---
def extract_metadata(full_text, metadata_fields, groq_api_key):
    """Extract metadata from document using LLM"""
    prompt = (
        f"Extract the following metadata from the text: {', '.join(metadata_fields)}.\n"
        "For each field, return the answer as a JSON object with keys as snake_case field names and values as strings or null. "
        "If a field is not found, set its value to null. Only output JSON.\n\n"
        f"Text:\n{full_text}"
    )
    llm_response = ask_llm(prompt, groq_api_key)
    if not llm_response:
        return {field: None for field in metadata_fields}
    try:
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
    answer_norm = answer_text.strip().lower()
    for item in line_boxes:
        if not isinstance(item, dict) or 'text' not in item or 'box' not in item:
            continue
        ocr_text_norm = item['text'].strip().lower()
        # Fuzzy match or substring match
        score = fuzz.token_set_ratio(answer_norm, ocr_text_norm)
        if score > 60 or answer_norm in ocr_text_norm:
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
    prompt = (
        "You are given OCR text extracted from a document.\n\n"
        f"OCR Text:\n{text}\n\n"
        f"Now answer this question: \"{question}\"\n\n"
        "Return only the **exact phrase** from the OCR text that answers the question. Do not paraphrase or explain."
    )
    answer = ask_llm(prompt, groq_api_key)
    if answer:
        highlighted_image = highlight_best_match(pil_image, line_boxes, answer)
        return answer, highlighted_image
    return None, pil_image

# --- MAIN STREAMLIT APP ---
def main():
    st.markdown('<h1 class="main-header">📄 Document Processing Pipeline</h1>', unsafe_allow_html=True)
    st.sidebar.title("🔧 Configuration")
    default_groq_api_key = (
        st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets
        else os.getenv("GROQ_API_KEY", "")
    )
    groq_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=default_groq_api_key,
        help="Enter your Groq API key to use the vision and text models"
    )
    st.sidebar.subheader("📤 Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document for processing"
    )
    st.sidebar.subheader("🏷️ Metadata Fields")
    default_fields = ["policy number", "policy start date", "policy end date", "policy holder name"]
    metadata_fields_input = st.sidebar.text_area(
        "Metadata fields to extract (one per line)",
        value="\n".join(default_fields),
        help="Enter the metadata fields you want to extract, one per line"
    )
    metadata_fields = [field.strip() for field in metadata_fields_input.split('\n') if field.strip()]

    # --- BUTTONS (only the required ones) ---
    headers_btn = st.button("Detect Headers & Segment")
    metadata_btn = st.button("Extract Metadata")
    qa_btn = st.button("Ask Question")

    # --- State variables ---
    pil_image = None
    preprocessed_image = None
    base64_image = None
    description = None
    header_structure = None
    document_segments = None
    metadata = None
    line_boxes = None

    # --- File and API Key Checks ---
    if not uploaded_file:
        st.info("👆 Please upload a PDF file using the sidebar to start processing.")
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
            - Multimodal LLM OCR
            - Object detection for text regions
            - Header detection & segmentation
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

    st.subheader(f"📋 Processing: {uploaded_file.name}")

    # --- Always process first page and OCR for downstream steps ---
    uploaded_file.seek(0)
    with st.spinner("Converting PDF to image..."):
        pil_image = pdf_to_image(uploaded_file)
    if pil_image is None:
        st.error("❌ Failed to convert PDF to image.")
        return
    st.subheader("📸 Original Document")
    st.image(pil_image, caption="Original PDF Page", use_container_width=True)

    with st.spinner("Preprocessing image for OCR..."):
        preprocessed_image = preprocess_image(pil_image)
    with st.expander("🔧 Preprocessed Image (for OCR)"):
        st.image(preprocessed_image, caption="Preprocessed Image", use_container_width=True)

    with st.spinner("Getting document description from vision model..."):
        base64_image = encode_image_to_base64(preprocessed_image)
        description = call_vision_model_for_description(base64_image, groq_api_key)
    if not description:
        st.error("❌ Could not get description from vision model.")
        return

    # Clean the OCR/LLM text before displaying
    cleaned_description = clean_llm_ocr_text(description)
    st.subheader("🤖 Vision Model Description")
    st.text_area("Extracted Text", cleaned_description, height=300)

    # --- HEADER DETECTION & SEGMENTATION ---
        # --- HEADER DETECTION & SEGMENTATION ---
    if headers_btn:
        with st.spinner("Detecting document headers..."):
            header_structure = detect_headers_recursive(description, groq_api_key)
        if header_structure:
            st.subheader("📋 Header Structure")
            # Pretty display of header hierarchy
            def render_headers(headers, level=0):
                if not isinstance(headers, dict):
                    return
                for header, content in headers.items():
                    indent = "&nbsp;" * (level * 6)
                    st.markdown(
                        f"{indent}<span style='font-size:{max(0.8, 1.4 - 0.15*level)}rem; color:#1f77b4; font-weight:bold;'>{header}</span>",
                        unsafe_allow_html=True
                    )
                    if isinstance(content, dict) and "subsections" in content:
                        render_headers(content["subsections"], level + 1)
            render_headers(header_structure)
            st.markdown("**Detected Headers (Beautified JSON):**")
            st.json(header_structure, expanded=False)

            # --- LLM-based Segment Processing ---
            def llm_process_segment(header, text, groq_api_key):
                """Use LLM to clean and prettify a segment's text for display."""
                prompt = (
                    f"Format and clean the following document section for human readability. "
                    f"Preserve tables, highlight key fields, and use Markdown if appropriate. "
                    f"Section Title: {header}\n\nSection Content:\n{text}\n\n"
                    "Return only the formatted section."
                )
                response = ask_llm(prompt, groq_api_key)
                return response if response else text

            document_segments = segment_document_by_headers(description, header_structure)
            st.subheader("📑 Document Segments")

            def render_segments_pretty(segments, groq_api_key, level=1):
                for header, content in segments.items():
                    # Combine the text for this segment
                    segment_text = "\n".join([line for line in content["text"] if line.strip()])
                    # Use LLM to prettify the segment text
                    pretty_text = llm_process_segment(header, segment_text, groq_api_key) if segment_text else ""
                    # Display as heading
                    st.markdown(f"{'#' * min(level,6)} {header}")
                    if pretty_text:
                        st.markdown(pretty_text, unsafe_allow_html=True)
                    if content["subsections"]:
                        render_segments_pretty(content["subsections"], groq_api_key, level + 1)

            render_segments_pretty(document_segments, groq_api_key)
        else:
            st.warning("No headers detected.")
            document_segments = {"Full Document": description.split('\n')}

    # --- METADATA EXTRACTION ---
    if metadata_btn:
        with st.spinner("Extracting metadata..."):
            metadata = extract_metadata(description, metadata_fields, groq_api_key)
        st.subheader("🏷️ Extracted Metadata")
        metadata_df = [{"Field": field, "Value": value} for field, value in metadata.items()]
        st.table(metadata_df)

    # --- Q&A ---
        # --- Q&A ---
    # Initialize session state for Q&A visibility
    if "show_qa" not in st.session_state:
        st.session_state.show_qa = False

    # When button is pressed, set flag
    if qa_btn:
        st.session_state.show_qa = True

    if st.session_state.show_qa:
        with st.spinner("Analyzing document structure..."):
            line_boxes = get_line_level_boxes(preprocessed_image, base64_image, groq_api_key)
        st.subheader("💬 Ask Questions About the Document")
        if "qa_answer" not in st.session_state:
            st.session_state.qa_answer = ""
        if "qa_highlighted_image" not in st.session_state:
            st.session_state.qa_highlighted_image = None

        question = st.text_input(
            "What would you like to know about this document?",
            placeholder="e.g., What is the policy number?"
        )
        if question and question.strip():
            with st.spinner(f"Finding answer to: {question}"):
                # answer, highlighted_image = ask_question_about_document(
                #     description, pil_image, line_boxes, question, groq_api_key
                # )
                answer, highlighted_image = ask_question_about_document(
                    description, preprocessed_image, line_boxes, question, groq_api_key
                )
            if answer and isinstance(answer, str) and answer.strip():
                st.session_state.qa_answer = answer
                st.session_state.qa_highlighted_image = highlighted_image
            else:
                st.session_state.qa_answer = ""
                st.session_state.qa_highlighted_image = None

        # Display retained answer and image if available
        if st.session_state.qa_answer:
            st.subheader("📝 Answer")
            st.success(f"**Answer:** {st.session_state.qa_answer}")
        if st.session_state.qa_highlighted_image is not None:
            st.subheader("🎯 Highlighted Document")
            st.image(st.session_state.qa_highlighted_image, caption=f"Answer highlighted for: {question}", use_container_width=True)
        elif question and not st.session_state.qa_answer:
            st.error("❌ Could not find an answer to your question.")

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

# ...existing code...

if __name__ == "__main__":
    main()