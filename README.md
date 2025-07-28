# Persona-Driven Document Intelligence System

## ✅ Approach

This system extracts structured heading information (Title, H1, H2, H3) from PDFs using a combination of layout features and semantic classification:

1. **Text Extraction:**  
   PDFs are parsed using `PyMuPDF` to extract text spans along with font size, position, and style metadata.

2. **Chunk Formation:**  
   Each page is segmented into text chunks based on font properties and visual cues.

3. **Semantic Labeling:**  
   Each chunk is classified using a lightweight multilingual ONNX model into one of the following: `Title`, `H1`, `H2`, `H3`, or `Other`.

4. **Hierarchy Building:**  
   Identified headings are structured into a clean hierarchy with page numbers. Title is treated separately, and all other headings are nested logically.

---

## 🧠 Models and Libraries Used

- **Model:**  
  A distilled ONNX transformer (~40MB) trained to classify heading levels from multilingual PDF content. The model is embedded in the Docker image.

- **Tokenizer:**  
  A tokenizer built using HuggingFace's `tokenizers` library is used to preprocess input text for ONNX inference.

- **Libraries:**  
  - `PyMuPDF (fitz)` – PDF parsing  
  - `onnxruntime` – ONNX model inference  
  - `tokenizers` – Fast tokenizer  
  - `huggingface_hub` – Tokenizer loading  
  - `numpy`, `tqdm`, `os`, `json`, `logging` – Supporting utilities

---

## 🐳 How to Build and Run

> 📌 Note: Your solution should run using the "Expected Execution" section. This build/run guide is for documentation only.

### 1. Build the Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
