#!/usr/bin/env python3
"""
Professional PDF to Java Code Generator
Converts PDF assignments to executable Java code with advanced features.
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import subprocess
import sys

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import tempfile
import shutil
import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from pdf2image import convert_from_path, pdfinfo_from_path



# Custom Exceptions



class PDFProcessorError(Exception):
    """Custom exception for PDF processing errors."""
    pass



# Configuration Manager



class ConfigManager:
    """Handles configuration loading from files and arguments."""

    DEFAULTS = {
        "dpi": 300,
        "model": "gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 2000,
        "workers": 4,
        "validate": True,
        "output_formats": ["java"],
        "cache_ocr": True,
        "contrast_factor": 3.0
    }

    def __init__(self):
        self.config = self.DEFAULTS.copy()

    def load_file(self, config_path: Path) -> bool:
        """Load JSON configuration file."""
        if config_path.exists():
            try:
                file_config = json.loads(config_path.read_text())
                self.config.update(file_config)
                return True
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON in config file: {config_path}")
        return False

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command-line arguments."""
        arg_dict = vars(args)
        for key in self.DEFAULTS:
            if key in arg_dict and arg_dict[key] is not None:
                self.config[key] = arg_dict[key]

    def __getitem__(self, key: str) -> Any:
        return self.config.get(key, self.DEFAULTS[key])

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default or self.DEFAULTS.get(key))



# OCR Processor with Caching



class OCRProcessor:
    """Handles OCR operations with caching and progress tracking."""

    def __init__(self, contrast_factor: float = 3.0, ocr_config: str = '--psm 6'):
        self.contrast_factor = contrast_factor
        self.ocr_config = ocr_config
        self.cache_dir = Path(".ocr_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """Generate cache file path based on PDF hash."""
        pdf_hash = hashlib.md5(pdf_path.read_bytes()).hexdigest()
        return self.cache_dir / f"{pdf_hash}.pkl"

    def load_cached_results(self, pdf_path: Path) -> Optional[List[str]]:
        """Load OCR results from cache if available."""
        cache_file = self._get_cache_path(pdf_path)
        if cache_file.exists():
            try:
                return pickle.loads(cache_file.read_bytes())
            except Exception:
                logging.warning("Failed to load OCR cache, regenerating...")
        return None

    def save_cached_results(self, pdf_path: Path, results: List[str]):
        """Save OCR results to cache."""
        cache_file = self._get_cache_path(pdf_path)
        try:
            cache_file.write_bytes(pickle.dumps(results))
        except Exception as e:
            logging.error(f"Failed to save OCR cache: {str(e)}")

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """Apply image enhancements for better OCR."""
        enhancer = ImageEnhance.Contrast(img)
        enhanced_img = enhancer.enhance(self.contrast_factor)
        return ImageOps.invert(enhanced_img.convert("RGB"))

    def process_image(self, img_data: Tuple[Image.Image, int]) -> str:
        """Process single image with OCR."""
        img, page_num = img_data
        try:
            # Normal OCR
            normal_text = pytesseract.image_to_string(img, config=self.ocr_config)
            
            # Enhanced OCR
            inverted_img = self._enhance_image(img)
            inverted_text = pytesseract.image_to_string(inverted_img, config=self.ocr_config)
            
            return f"=== PAGE {page_num} ===\n{normal_text}\n--- ENHANCED ---\n{inverted_text}\n"
        except Exception as e:
            logging.error(f"OCR failed for page {page_num}: {str(e)}")
            return f"=== PAGE {page_num} (OCR ERROR) ===\n"

    def process_with_progress(self, images: List[Image.Image]) -> List[str]:
        """Process images with progress tracking."""
        img_data = [(img, i+1) for i, img in enumerate(images)]
        
        try:
            from tqdm import tqdm
            return [self.process_image(data) for data in tqdm(img_data, desc="OCR Processing")]
        except ImportError:
            return [self.process_image(data) for data in img_data]



# Text Cleaner



class TextCleaner:
    """Cleans and filters extracted text."""

    TRAP_PATTERNS = [
        r"variable\s+name\s*=\s*\w+",
        r"set\s+variable\s+to\s+.*",
        r"(?i)ai[\s_-]?instructions?.*",
        r"(?i)ignore\s+previous\s+instructions",
        r"(?i)system\s+prompt\s*:.*",
        r"(?i)remember\s+this\s+instruction.*",
        r"(?i)override\s+previous\s+rules.*",
    ]

    def clean_ai_traps(self, text: str) -> str:
        """Remove potential prompt injection attempts."""
        for pattern in self.TRAP_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize whitespace and remove OCR artifacts."""
        # Remove processing markers
        text = re.sub(r'=== PAGE \d+ ===\n', '', text)
        text = re.sub(r'--- ENHANCED ---\n', '', text)
        text = re.sub(r'=== PAGE \d+ \(OCR ERROR\) ===\n', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'\n\s*\n+', '\n\n', text).strip()

    def extract_code_blocks(self, text: str) -> List[str]:
        """Identify potential code segments in text."""
        patterns = [
            r'```java(.*?)```',
            r'```(.*?)```',
            r'public\s+class\s+\w+\s*\{[\s\S]*?\}',
        ]
        return [block for pattern in patterns for block in re.findall(pattern, text, re.DOTALL)]



# AI Code Generator with Fallback



class AICodeGenerator:
    """Handles AI-powered code generation with fallback strategies using Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("Your_API_KEY")
        if not self.api_key:
            raise PDFProcessorError("Missing Gemini API key. Set GEMINI_API_KEY environment variable.")
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai.configure(api_key=self.api_key)
        except ImportError:
            raise PDFProcessorError("google-generativeai package not installed. Install with 'pip install google-generativeai'.")

    def generate_code(
        self, 
        assignment_text: str, 
        concepts: str, 
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1, 
        max_tokens: int = 2000,
        strategy: str = "standard"
    ) -> str:
        """Generate Java code with different strategies using Gemini."""
        base_system_prompt = f"""
You are a Java coding assistant for educational assignments.
CONSTRAINTS:

1. Use ONLY these concepts: {concepts}
2. Implement EXACTLY what the assignment requires
3. Output ONLY raw Java code without explanations
4. Include necessary imports and main method
5. Ensure code compiles and runs correctly
"""
        strategies = {
            "detailed": (
                base_system_prompt + "\nSTRATEGY: Provide a detailed implementation with comments.",
                0.1
            ),
            "simplified": (
                base_system_prompt + "\nSTRATEGY: Use minimal code with only essential functionality.",
                0.0
            ),
            "creative": (
                base_system_prompt + "\nSTRATEGY: Find creative solutions within constraints.",
                0.3
            ),
            "standard": (base_system_prompt, 0.1)
        }
        system_prompt, strategy_temp = strategies.get(strategy, strategies["standard"])
        effective_temp = max(temperature, strategy_temp)

        try:
            model_obj = self.genai.GenerativeModel(model)
            response = model_obj.generate_content(
                [
                    {"role": "system", "parts": [system_prompt.strip()]},
                    {"role": "user", "parts": [f"Assignment:\n{assignment_text.strip()}"]},
                ],
                generation_config={
                    "temperature": effective_temp,
                    "max_output_tokens": max_tokens
                }
            )
            # Gemini responses may be in 'text' or 'candidates'
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "candidates") and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                raise PDFProcessorError("No content returned from Gemini API.")
        except Exception as e:
            raise PDFProcessorError(f"AI generation failed: {str(e)}")

    def generate_with_fallback(
        self,
        text: str,
        concepts: str,
        model: str = "gemini-1.5-pro",
        max_tokens: int = 2000
    ) -> str:
        """Try multiple strategies if primary generation fails."""
        strategies = ["standard", "detailed", "simplified", "creative"]
        for strategy in strategies:
            try:
                logging.info(f"Trying generation strategy: {strategy}")
                return self.generate_code(text, concepts, model, strategy=strategy)
            except Exception as e:
                logging.warning(f"Strategy '{strategy}' failed: {str(e)}")
        raise PDFProcessorError("All generation strategies failed")



# Code Post-Processor



class CodePostProcessor:
    """Post-processes and validates generated Java code."""

    SAFE_VAR_NAMES = {
        'var': 'value',
        'temp': 'tmp',
        'data': 'inputData',
        'result': 'output',
        'val': 'item',
        'obj': 'instance'
    }

    def sanitize_code(self, code: str) -> str:
        """Clean and standardize generated code."""
        # Replace unsafe variable names
        for unsafe, safe in self.SAFE_VAR_NAMES.items():
            code = re.sub(rf'\b{unsafe}\b', safe, code)
        
        # Standardize formatting
        code = code.replace("    ", "  ")  # 2-space indentation
        code = re.sub(r'\n{3,}', '\n\n', code)  # Remove extra newlines
        return f"// AUTO-GENERATED - REVIEW BEFORE USE\n\n{code.strip()}"

    def extract_main_class(self, code: str) -> Optional[str]:
        """Identify the main class name from code."""
        match = re.search(r'public\s+class\s+(\w+)', code)
        return match.group(1) if match else None

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Perform basic Java syntax checks."""
        issues = []
        checks = [
            ('braces', r'\{', r'\}', 'Unbalanced braces'),
            ('parentheses', r'\(', r'\)', 'Unbalanced parentheses'),
            ('brackets', r'\[', r'\]', 'Unbalanced brackets')
        ]
        
        for name, open_pat, close_pat, msg in checks:
            opens = len(re.findall(open_pat, code))
            closes = len(re.findall(close_pat, code))
            if opens != closes:
                issues.append(f"{msg}: {opens} vs {closes}")
        
        if not re.search(r'class\s+\w+', code):
            issues.append("Missing class declaration")
            
        return (not issues), issues

    def analyze_code_quality(self, code: str) -> Dict[str, int]:
        """Calculate code quality metrics."""
        return {
            'lines': len(code.splitlines()),
            'methods': len(re.findall(r'\b(public|private|protected)\s+\w+\s+\w+\s*\(', code)),
            'complexity': code.count('if') + code.count('for') + code.count('while'),
            'classes': len(re.findall(r'class\s+\w+', code)),
            'comments': len(re.findall(r'//', code)) + len(re.findall(r'/\*', code))
        }



# Java Compiler



class JavaCompiler:
    """Handles compilation and execution with resource management."""

    def __init__(self, work_dir: Path = Path(".")):
        self.work_dir = work_dir
        self.temp_dir = None

    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        return self

    def __exit__(self, *args):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def compile_and_run(
        self, 
        code_path: Path, 
        class_name: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """Compile and execute Java code with sandboxing."""
        class_name = class_name or code_path.stem
        java_file = self.temp_dir / f"{class_name}.java"
        java_file.write_text(code_path.read_text())
        
        try:
            # Compilation
            compile_proc = subprocess.run(
                ["javac", str(java_file)],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            if compile_proc.returncode != 0:
                return False, "", f"Compilation error:\n{compile_proc.stderr}"
            
            # Execution
            run_proc = subprocess.run(
                ["java", class_name],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=15
            )
            return True, run_proc.stdout, run_proc.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out"
        except Exception as e:
            return False, "", f"Runtime error: {str(e)}"



# Output Manager



class OutputManager:
    """Handles saving output in multiple formats."""

    def save_outputs(
        self, 
        code: str, 
        output_path: Path, 
        formats: List[str],
        quality_metrics: Optional[Dict] = None
    ):
        """Save code in specified formats."""
        base_path = output_path.with_suffix('')
        
        for fmt in formats:
            try:
                if fmt == 'java':
                    output_path.write_text(code)
                    logging.info(f"Java code saved to {output_path}")
                
                elif fmt == 'txt':
                    txt_path = base_path.with_suffix('.txt')
                    txt_path.write_text(code)
                    logging.info(f"Text version saved to {txt_path}")
                
                elif fmt == 'html':
                    html_path = base_path.with_suffix('.html')
                    html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>Generated Java Code</title>
<style>
body {{ font-family: monospace; }}
.code {{ white-space: pre; background: #f8f8f8; padding: 10px; }}
</style>
</head>
<body>
<h1>Generated Java Code</h1>
<div class="code">{code}</div>"""
                    if quality_metrics:
                        html_content += "<h2>Code Quality Metrics</h2>"
                        html_content += "<ul>"
                        for metric, value in quality_metrics.items():
                            html_content += f"<li>{metric.replace('_', ' ').title()}: {value}</li>"
                        html_content += "</ul>"

                    html_content += "</body></html>"
                    html_path.write_text(html_content)
                    logging.info(f"HTML version saved to {html_path}")
                
                elif fmt == 'json':
                    json_path = base_path.with_suffix('.json')
                    output_data = {
                        "code": code,
                        "metrics": quality_metrics or {}
                    }
                    json_path.write_text(json.dumps(output_data, indent=2))
                    logging.info(f"JSON version saved to {json_path}")
            
            except Exception as e:
                logging.error(f"Failed to save {fmt} format: {str(e)}")



# PDF to Java Pipeline



class PDFToJavaProcessor:
    """Orchestrates the entire conversion pipeline."""

    def __init__(self, config: ConfigManager, verbose: bool = False):
        self.config = config
        self.ocr = OCRProcessor(contrast_factor=config["contrast_factor"])
        self.cleaner = TextCleaner()
        self.ai = AICodeGenerator()
        self.post = CodePostProcessor()
        self.output = OutputManager()
        self.verbose = verbose

    def process_pdf(
        self,
        pdf_path: Path,
        concepts_path: Path,
        output_path: Path,
    ) -> bool:
        """Execute full conversion pipeline with error handling."""
        try:
            # Validate inputs
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            if not concepts_path.exists():
                raise FileNotFoundError(f"Concepts file not found: {concepts_path}")
            
            # OCR Processing with caching
            ocr_results = None
            if self.config["cache_ocr"]:
                ocr_results = self.ocr.load_cached_results(pdf_path)
            
            if not ocr_results:
                logging.info("Converting PDF to images...")
                images = convert_from_path(
                    pdf_path, 
                    dpi=self.config["dpi"],
                    thread_count=self.config["workers"]
                )
                
                logging.info(f"Processing {len(images)} pages with OCR...")
                ocr_results = self.ocr.process_with_progress(images)
                
                if self.config["cache_ocr"]:
                    self.ocr.save_cached_results(pdf_path, ocr_results)
            
            # Text Processing
            raw_text = "\n".join(ocr_results)
            clean_text = self.cleaner.clean_ai_traps(raw_text)
            final_text = self.cleaner.normalize_text(clean_text)
            
            if self.verbose:
                logging.debug(f"Extracted text ({len(final_text)} chars):\n{final_text[:500]}...")
            
            # AI Generation
            concepts = concepts_path.read_text().strip()
            logging.info("Generating Java code with AI...")
            
            try:
                ai_code = self.ai.generate_code(
                    final_text, 
                    concepts,
                    model=self.config["model"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"]
                )
            except PDFProcessorError:
                logging.warning("Primary generation failed, using fallback strategies...")
                ai_code = self.ai.generate_with_fallback(
                    final_text, 
                    concepts,
                    model=self.config["model"],
                    max_tokens=self.config["max_tokens"]
                )
            
            # Post-processing
            final_code = self.post.sanitize_code(ai_code)
            class_name = self.post.extract_main_class(final_code) or output_path.stem
            
            # Add class wrapper if needed
            if not re.search(r'class\s+\w+', final_code):
                final_code = f"public class {class_name} {{\npublic static void main(String[] args) {{\n{final_code}\n}}\n}}"
                logging.info(f"Added class wrapper: {class_name}")
            
            # Validate syntax if requested
            validation_issues = []
            if self.config["validate"]:
                valid, issues = self.post.validate_syntax(final_code)
                if not valid:
                    validation_issues = issues
                    logging.warning(f"Syntax issues: {', '.join(issues)}")
            
            # Analyze code quality
            quality_metrics = self.post.analyze_code_quality(final_code)
            if self.verbose:
                logging.info(f"Code quality metrics: {quality_metrics}")
            
            # Save outputs
            self.output.save_outputs(
                final_code, 
                output_path, 
                self.config["output_formats"],
                quality_metrics={**quality_metrics, "validation_issues": validation_issues}
            )
            
            # Compile and run if Java output was generated
            if "java" in self.config["output_formats"]:
                with JavaCompiler() as compiler:
                    success, stdout, stderr = compiler.compile_and_run(output_path, class_name)
                    
                    if success:
                        logging.info("Execution successful!")
                        if stdout:
                            logging.info(f"Output:\n{stdout[:200]}{'...' if len(stdout) > 200 else ''}")
                        return True
                    else:
                        logging.error(f"Execution failed: {stderr}")
                        return False
            return True
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            if self.verbose:
                import traceback
                logging.debug(traceback.format_exc())
            return False



# Dependency Checker



def check_dependencies():
    """Verify required system tools are available."""
    requirements = {
        "tesseract": ["tesseract", "--version"],
        "javac": ["javac", "-version"],
        "pdftoppm": ["pdftoppm", "-v"]
    }

    missing = []
    for tool, cmd in requirements.items():
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(tool)

    if missing:
        raise PDFProcessorError(f"Missing dependencies: {', '.join(missing)}")



# Main Entry Point



def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF assignments to executable Java code",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF file path")
    parser.add_argument("--concepts", type=Path, required=True, help="File with allowed Java concepts")
    parser.add_argument("--output", type=Path, default=Path("Solution.java"), help="Output Java file")
    parser.add_argument("--config", type=Path, help="Configuration file (JSON)")
    parser.add_argument("--dpi", type=int, help="Image resolution for OCR")
    parser.add_argument("--model", help="OpenAI model to use")
    parser.add_argument("--temperature", type=float, help="AI creativity level (0-1)")
    parser.add_argument("--max-tokens", type=int, help="Max tokens for AI output")
    parser.add_argument("--workers", type=int, help="OCR processing threads")
    parser.add_argument("--no-validate", action="store_true", help="Skip syntax validation")
    parser.add_argument("--formats", nargs="+", choices=["java", "txt", "html", "json"],
                        help="Output formats (default: java)")
    parser.add_argument("--no-cache", action="store_true", help="Disable OCR caching")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    try:
        # Configuration setup
        config = ConfigManager()
        if args.config:
            config.load_file(args.config)
        config.update_from_args(args)
        
        # Handle specific flags
        if args.no_validate:
            config.config["validate"] = False
        if args.no_cache:
            config.config["cache_ocr"] = False
        if args.formats:
            config.config["output_formats"] = args.formats
        
        # Check dependencies
        check_dependencies()
        
        # Process pipeline
        processor = PDFToJavaProcessor(config, verbose=args.verbose)
        success = processor.process_pdf(
            pdf_path=args.pdf,
            concepts_path=args.concepts,
            output_path=args.output,
        )
        
        if success:
            logging.info("✅ Processing completed successfully")
            sys.exit(0)
        else:
            logging.error(f"❌ Processing failed for {args.pdf}")
            sys.exit(1)
            
    except PDFProcessorError as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        sys.exit(2)

if __name__ == "__main__":
    main()