# Arabic OCR PDF Text Extraction

A comprehensive solution to extract Arabic text from PDF files containing images using OCR and AI methods while preserving the exact document structure including pages, spaces, lines, and other characters.

## Features

- **PDF Processing**: Extract images from PDF files (supports both scanned PDFs and PDFs with embedded images)
- **Arabic OCR**: Optimized for Arabic text recognition using multiple OCR engines
- **Structure Preservation**: Maintains exact layout including:
  - Page breaks
  - Line breaks
  - Horizontal and vertical spacing
  - Special characters
  - Text alignment (RTL for Arabic)
- **Multiple OCR Engines**:
  - Tesseract OCR with Arabic language support
  - EasyOCR for Arabic text (with fallback support)
- **Image Preprocessing**: Improves OCR accuracy with:
  - Deskewing (corrects tilted scans)
  - Noise removal
  - Contrast enhancement
- **Multiple Output Formats**:
  - Plain text (.txt)
  - Markdown (.md)
  - Word document (.docx)
- **Batch Processing**: Process multiple PDF files at once
- **Confidence Scoring**: Reports OCR confidence for quality assessment
- **Progress Indicators**: Visual progress bars for long operations
- **Comprehensive Logging**: Detailed logs for debugging

## Installation

### System Requirements

1. **Python 3.8+**

2. **Tesseract OCR** (required for Tesseract engine):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install tesseract-ocr tesseract-ocr-ara
   
   # macOS (using Homebrew)
   brew install tesseract tesseract-lang
   
   # Windows
   # Download installer from https://github.com/UB-Mannheim/tesseract/wiki
   # Make sure to select Arabic language during installation
   ```

3. **Poppler** (for pdf2image, alternative to PyMuPDF):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install poppler-utils
   
   # macOS
   brew install poppler
   
   # Windows
   # Download from http://blog.alivate.com.au/poppler-windows/
   ```

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install PyMuPDF pytesseract easyocr Pillow opencv-python numpy python-docx tqdm
```

## Quick Start

### Basic Usage

```bash
# Process a single PDF file
python arabic_ocr.py document.pdf

# Specify output directory
python arabic_ocr.py document.pdf --output-dir ./extracted/

# Choose specific output formats
python arabic_ocr.py document.pdf --formats txt md

# Use EasyOCR instead of Tesseract
python arabic_ocr.py document.pdf --engine easyocr
```

### Batch Processing

```bash
# Process all PDFs in a directory
python arabic_ocr.py ./pdf_folder/ --batch

# Batch with specific output directory
python arabic_ocr.py ./pdf_folder/ --batch --output-dir ./all_extracted/
```

### Python API

```python
from arabic_ocr import ArabicOCR

# Initialize OCR processor
ocr = ArabicOCR(
    ocr_engine="tesseract",  # or "easyocr"
    use_fallback=True,       # Use secondary engine if primary has low confidence
    min_confidence=60,       # Minimum confidence threshold
)

# Process a single PDF
result = ocr.process_pdf(
    "document.pdf",
    output_dir="./output/",
    formats=["txt", "md", "docx"]
)

# Access results
print(f"Processed {result.total_pages} pages")
print(f"Average confidence: {result.average_confidence:.1f}%")

for page in result.pages:
    print(f"Page {page.page_num}: {page.structured_text[:100]}...")
```

## Command Line Options

```
usage: arabic_ocr.py [-h] [-o OUTPUT_DIR] [-f {txt,md,docx} [{txt,md,docx} ...]]
                     [-e {tesseract,easyocr}] [--batch] [--no-fallback]
                     [--min-confidence MIN_CONFIDENCE] [-v]
                     input

Extract Arabic text from PDF files using OCR

positional arguments:
  input                 Input PDF file or directory (with --batch)

optional arguments:
  -h, --help            show this help message and exit
  -o, --output-dir      Output directory for extracted text
  -f, --formats         Output formats (default: txt md docx)
  -e, --engine          OCR engine to use (default: tesseract)
  --batch               Process all PDF files in the input directory
  --no-fallback         Disable fallback to secondary OCR engine
  --min-confidence      Minimum confidence threshold (0-100, default: 60)
  -v, --verbose         Enable verbose output
```

## Configuration

The `config.py` file contains all configurable settings:

### OCR Settings

```python
OCR_CONFIG = {
    "primary_engine": "tesseract",  # Primary OCR engine
    "use_fallback": True,           # Use fallback if primary fails
    "min_confidence": 60,           # Minimum confidence to accept
    
    "tesseract": {
        "lang": "ara",              # Language: "ara", "ara+eng"
        "psm": 3,                   # Page segmentation mode
        "oem": 1,                   # OCR engine mode
    },
    
    "easyocr": {
        "languages": ["ar", "en"],  # Languages to use
        "gpu": False,               # Use GPU if available
    },
}
```

### Preprocessing Settings

```python
PREPROCESSING_CONFIG = {
    "enabled": True,
    "deskew": True,           # Correct skewed images
    "denoise": True,          # Remove noise
    "enhance_contrast": True, # Improve contrast
    "binarize": False,        # Convert to black/white
}
```

### Output Settings

```python
OUTPUT_CONFIG = {
    "formats": ["txt", "md", "docx"],
    
    "docx": {
        "font_name": "Arial",     # Font with Arabic support
        "font_size": 12,
        "rtl_direction": True,    # Right-to-left text
    },
}
```

## Output Format Details

### Plain Text (.txt)
- UTF-8 encoding
- Page separators with page numbers
- Preserved line breaks and spacing

### Markdown (.md)
- RTL text wrapping with `<div dir="rtl">`
- Page breaks as horizontal rules
- Document metadata (pages, confidence)

### Word Document (.docx)
- Arabic font support
- RTL paragraph alignment
- Page breaks between pages

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   - Ensure Tesseract is installed and in PATH
   - Set `TESSDATA_PREFIX` environment variable if needed

2. **Arabic language data missing**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr-ara
   
   # Or download manually from:
   # https://github.com/tesseract-ocr/tessdata
   ```

3. **Low OCR accuracy**
   - Try using EasyOCR with `--engine easyocr`
   - Enable all preprocessing options in config
   - Use higher DPI PDFs when possible

4. **Memory issues with large PDFs**
   - Process in smaller batches
   - Reduce DPI in `PDF_CONFIG`
   - Set `max_pages` in config

## Dependencies

- **pdf2image** or **PyMuPDF**: PDF to image conversion
- **pytesseract**: Tesseract OCR wrapper
- **easyocr**: Alternative OCR engine
- **Pillow**: Image processing
- **opencv-python**: Image preprocessing
- **numpy**: Numerical operations
- **python-docx**: Word document generation
- **tqdm**: Progress bars

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.