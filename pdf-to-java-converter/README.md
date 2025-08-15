# PDF to Java Code Converter V1.1

Convert PDF programming assignments to executable Java code using OCR and AI.

##1.1 update
Added error handaling for API Integration and improved response parsing
Adjusted OCR for finer accuracy 
Added syntax validition for better output

## Features
PDF text extraction via OCR
Java code generation powered by google gemeni
Caching for faster re-processing 
Prompt Injection detection
Rotating API key integration
Multiple Download formats

## Requirements

Python 3.8+
Java JDK (for compilation testing)
Tesseract OCR
Poppler Utils (for PDF processing)
Google Gemini API Key(s)

## Installation

On Ubuntu/Debian:
bashsudo apt update
sudo apt install tesseract-ocr poppler-utils default-jdk
pip install -r requirements.txt
On macOS:
bashbrew install tesseract poppler openjdk
pip install -r requirements.txt

Windows not supported, this program was intended for MacOS and Linux Users.

## Usage Examples

Basic Usage

bashpython pdf_to_java_generator.py --pdf assignment.pdf --concepts allowed_concepts.txt

Advanced Options

  pdf_to_java_generator.py \
 
  --pdf complex_assignment.pdf \
  
  --concepts java_basics.txt \
  
  --output MyJavaCode.java \
  
  --formats java html json \
  
  --verbose

## Using Configuration File

bashpython pdf_to_java_generator.py \

  --pdf assignment.pdf \
  
  --concepts concepts.txt \
  
  --config configs/default_config.json
 
## Configuration

Create a config.json file:

json{

  "dpi": 300,
  
  "model": "gemini-1.5-pro",
  
  "temperature": 0.1,
  
  "max_tokens": 4000,
  
  "output_formats": ["java", "html"],
  
  "cache_ocr": true,
  
  "validate": true

}

## Troubleshooting Common Issues:

"No API keys found" - Set the GEMINI_API_KEY environment variable

Poor OCR quality - Increase DPI or check PDF quality

## Contributing

This is a personal learning project, any critisism or help is welcome and appreciated.

Fork the repo

Create a feature branch
Submit a pull request
