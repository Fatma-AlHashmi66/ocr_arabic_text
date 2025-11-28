"""
Configuration settings for Arabic OCR PDF extraction.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# OCR Engine Settings
OCR_CONFIG = {
    # Primary OCR engine: 'tesseract' or 'easyocr'
    "primary_engine": "tesseract",
    
    # Use fallback engine if primary fails or has low confidence
    "use_fallback": True,
    
    # Minimum confidence score (0-100) to accept OCR result
    "min_confidence": 60,
    
    # Tesseract-specific settings
    "tesseract": {
        # Language codes: 'ara' for Arabic, 'ara+eng' for Arabic and English
        "lang": "ara",
        # Page segmentation mode (PSM)
        # 3 = Fully automatic page segmentation, but no OSD
        # 6 = Assume a single uniform block of text
        # 11 = Sparse text - find as much text as possible
        "psm": 3,
        # OCR Engine Mode (OEM)
        # 1 = Neural nets LSTM engine only
        # 3 = Default, based on what is available
        "oem": 1,
        # Custom config string
        "config": "--dpi 300",
    },
    
    # EasyOCR-specific settings
    "easyocr": {
        # Languages to use
        "languages": ["ar", "en"],
        # Use GPU if available
        "gpu": False,
        # Text detection confidence threshold
        "text_threshold": 0.7,
        # Link threshold for character grouping
        "link_threshold": 0.4,
        # Low text confidence threshold
        "low_text": 0.4,
    },
}

# Image Preprocessing Settings
PREPROCESSING_CONFIG = {
    # Enable preprocessing
    "enabled": True,
    
    # Deskewing
    "deskew": True,
    
    # Noise removal
    "denoise": True,
    "denoise_strength": 10,
    
    # Contrast enhancement
    "enhance_contrast": True,
    "contrast_factor": 1.5,
    
    # Binarization (convert to black and white)
    "binarize": False,
    "binarize_threshold": 128,
    
    # Upscaling for low-resolution images
    "upscale": True,
    "upscale_factor": 2,
    "upscale_min_dpi": 150,
    
    # Target DPI for processing
    "target_dpi": 300,
}

# PDF Processing Settings
PDF_CONFIG = {
    # DPI for image extraction
    "dpi": 300,
    
    # Image format for extracted pages
    "image_format": "png",
    
    # Use grayscale for processing (faster, usually better for OCR)
    "grayscale": True,
    
    # Thread count for parallel processing (-1 for auto)
    "thread_count": -1,
    
    # Maximum pages to process (None for all)
    "max_pages": None,
}

# Structure Preservation Settings
STRUCTURE_CONFIG = {
    # Page separator in output
    "page_separator": "\n--- Page {page_num} ---\n",
    
    # Preserve spacing ratios
    "preserve_spacing": True,
    
    # Minimum horizontal space ratio to insert extra spaces
    "horizontal_space_threshold": 2.0,
    
    # Minimum vertical space ratio to insert extra lines
    "vertical_space_threshold": 1.5,
    
    # Right-to-left text direction
    "rtl": True,
    
    # Detect and preserve tables
    "detect_tables": True,
    
    # Detect and preserve columns
    "detect_columns": True,
}

# Output Format Settings
OUTPUT_CONFIG = {
    # Default output formats
    "formats": ["txt", "md", "docx"],
    
    # Text file settings
    "txt": {
        "encoding": "utf-8",
        "include_page_numbers": True,
    },
    
    # Markdown settings
    "md": {
        "page_break_style": "hr",  # 'hr' for horizontal rule, 'header' for page headers
        "include_toc": False,  # Include table of contents
    },
    
    # Word document settings
    "docx": {
        "font_name": "Arial",  # Good Arabic support: Arial, Simplified Arabic, Traditional Arabic
        "font_size": 12,
        "rtl_direction": True,
        "include_page_breaks": True,
    },
}

# Logging Settings
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "ocr_arabic.log",
    "console": True,
}

# Batch Processing Settings
BATCH_CONFIG = {
    # Continue on error (process remaining files even if one fails)
    "continue_on_error": True,
    
    # Number of parallel workers (1 for sequential)
    "workers": 1,
    
    # Show progress bar
    "show_progress": True,
}
