#!/usr/bin/env python3
"""
Enhanced Professional PDF to Java Code Generator
Converts PDF assignments to executable Java code with improved error handling and features.

Key Improvements:
1. Fixed Gemini API integration with proper model instantiation
2. Enhanced error handling and logging
3. Improved OCR processing with better image preprocessing
4. More robust code validation and post-processing
5. Better resource management and cleanup
6. Enhanced configuration validation
7. Improved text extraction and cleaning
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
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

# Third-party imports with better error handling
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageOps, ImageFilter
    from pdf2image import convert_from_path, pdfinfo_from_path
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install pytesseract pillow pdf2image")
    sys.exit(1)

# Optional imports
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# Custom Exceptions
class PDFProcessorError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class APIError(Exception):
    """Exception for API-related errors."""
    pass


# Configuration Manager with Validation
class ConfigManager:
    """Enhanced configuration manager with validation."""

    DEFAULTS = {
        "dpi": 300,
        "model": "gemini-1.5-pro",
        "temperature": 0.1,
        "max_tokens": 4000,
        "workers": 4,
        "validate": True,
        "output_formats": ["java"],
        "cache_ocr": True,
        "contrast_factor": 2.5,
        "max_pages": 50,
        "timeout": 60,
        "retry_attempts": 3,
        "retry_delay": 2.0
    }

    VALID_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro"
    ]

    VALID_FORMATS = ["java", "txt", "html", "json", "markdown"]

    def __init__(self):
        self.config = self.DEFAULTS.copy()

    def load_file(self, config_path: Path) -> bool:
        """Load and validate JSON configuration file."""
        if not config_path.exists():
            logging.warning(f"Config file not found: {config_path}")
            return False
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # Validate configuration
            self._validate_config(file_config)
            self.config.update(file_config)
            logging.info(f"Configuration loaded from {config_path}")
            return True
            
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            return False

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration values."""
        if "model" in config and config["model"] not in self.VALID_MODELS:
            raise ValueError(f"Invalid model: {config['model']}. Must be one of {self.VALID_MODELS}")
        
        if "output_formats" in config:
            invalid_formats = set(config["output_formats"]) - set(self.VALID_FORMATS)
            if invalid_formats:
                raise ValueError(f"Invalid output formats: {invalid_formats}")
        
        # Range validations
        validations = [
            ("dpi", 150, 600),
            ("temperature", 0.0, 2.0),
            ("max_tokens", 1000, 8000),
            ("workers", 1, 16),
            ("contrast_factor", 1.0, 5.0),
            ("max_pages", 1, 200)
        ]
        
        for key, min_val, max_val in validations:
            if key in config:
                val = config[key]
                if not (min_val <= val <= max_val):
                    raise ValueError(f"{key} must be between {min_val} and {max_val}, got {val}")

    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command-line arguments."""
        arg_mapping = {
            'dpi': 'dpi',
            'model': 'model', 
            'temperature': 'temperature',
            'max_tokens': 'max_tokens',
            'workers': 'workers',
            'no_validate': lambda x: not x,
            'no_cache': lambda x: not x,
            'formats': 'output_formats'
        }
        
        for arg_key, config_key in arg_mapping.items():
            if hasattr(args, arg_key):
                value = getattr(args, arg_key)
                if value is not None:
                    if callable(config_key):
                        if arg_key == 'no_validate':
                            self.config['validate'] = not value
                        elif arg_key == 'no_cache':
                            self.config['cache_ocr'] = not value
                    else:
                        self.config[config_key] = value

    def __getitem__(self, key: str) -> Any:
        return self.config.get(key, self.DEFAULTS[key])

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default or self.DEFAULTS.get(key))


# Enhanced OCR Processor
class OCRProcessor:
    """Enhanced OCR processor with better image preprocessing and error handling."""

    def __init__(self, contrast_factor: float = 2.5, dpi: int = 300):
        self.contrast_factor = contrast_factor
        self.dpi = dpi
        self.cache_dir = Path(".ocr_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # OCR configurations for different scenarios
        self.ocr_configs = {
            'default': '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}"\'-+=_@#$%^&*/<>|\\~ \t\n',
            'code': '--psm 6 -c preserve_interword_spaces=1',
            'single_block': '--psm 8',
            'single_line': '--psm 7'
        }

    def _get_cache_path(self, pdf_path: Path) -> Path:
        """Generate cache file path based on PDF hash and settings."""
        pdf_content = pdf_path.read_bytes()
        settings_str = f"{self.contrast_factor}_{self.dpi}"
        combined_hash = hashlib.md5(pdf_content + settings_str.encode()).hexdigest()
        return self.cache_dir / f"{combined_hash}.pkl"

    def load_cached_results(self, pdf_path: Path) -> Optional[List[str]]:
        """Load OCR results from cache if available."""
        cache_file = self._get_cache_path(pdf_path)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if isinstance(cached_data, dict) and 'results' in cached_data:
                        logging.info("Loaded OCR results from cache")
                        return cached_data['results']
            except Exception as e:
                logging.warning(f"Failed to load OCR cache: {e}")
                # Remove corrupted cache
                try:
                    cache_file.unlink()
                except:
                    pass
        return None

    def save_cached_results(self, pdf_path: Path, results: List[str]):
        """Save OCR results to cache with metadata."""
        cache_file = self._get_cache_path(pdf_path)
        try:
            cache_data = {
                'results': results,
                'timestamp': time.time(),
                'settings': {
                    'contrast_factor': self.contrast_factor,
                    'dpi': self.dpi
                }
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            logging.debug(f"OCR results cached to {cache_file}")
        except Exception as e:
            logging.error(f"Failed to save OCR cache: {e}")

    def _preprocess_image(self, img: Image.Image) -> List[Image.Image]:
        """Apply multiple preprocessing techniques for better OCR."""
        processed_images = []
        
        # Original image
        processed_images.append(img)
        
        # Convert to grayscale if not already
        if img.mode != 'L':
            gray_img = img.convert('L')
            processed_images.append(gray_img)
        else:
            gray_img = img
        
        # Enhanced contrast
        enhancer = ImageEnhance.Contrast(gray_img)
        contrast_img = enhancer.enhance(self.contrast_factor)
        processed_images.append(contrast_img)
        
        # Sharpened version
        sharpened = contrast_img.filter(ImageFilter.SHARPEN)
        processed_images.append(sharpened)
        
        # Inverted for dark backgrounds
        inverted = ImageOps.invert(contrast_img)
        processed_images.append(inverted)
        
        return processed_images

    def _extract_text_from_image(self, img: Image.Image, config_key: str = 'default') -> str:
        """Extract text using specific OCR configuration."""
        try:
            config = self.ocr_configs.get(config_key, self.ocr_configs['default'])
            return pytesseract.image_to_string(img, config=config).strip()
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return ""

    def process_image(self, img_data: Tuple[Image.Image, int]) -> str:
        """Process single image with multiple preprocessing approaches."""
        img, page_num = img_data
        
        try:
            processed_images = self._preprocess_image(img)
            best_text = ""
            best_length = 0
            
            # Try different preprocessing approaches
            for i, processed_img in enumerate(processed_images):
                for config_key in ['default', 'code']:
                    try:
                        text = self._extract_text_from_image(processed_img, config_key)
                        # Select the result with most content
                        if len(text) > best_length:
                            best_text = text
                            best_length = len(text)
                    except Exception as e:
                        logging.debug(f"OCR attempt {i}-{config_key} failed: {e}")
                        continue
            
            return f"=== PAGE {page_num} ===\n{best_text}\n"
            
        except Exception as e:
            logging.error(f"Failed to process page {page_num}: {e}")
            return f"=== PAGE {page_num} (ERROR) ===\n"

    def process_with_progress(self, images: List[Image.Image]) -> List[str]:
        """Process images with progress tracking and parallel processing."""
        img_data = [(img, i + 1) for i, img in enumerate(images)]
        
        if HAS_TQDM:
            progress_bar = tqdm(total=len(img_data), desc="OCR Processing")
        
        results = []
        
        # Process images in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(4, len(img_data))) as executor:
            future_to_data = {executor.submit(self.process_image, data): data for data in img_data}
            
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                    if HAS_TQDM:
                        progress_bar.update(1)
                except Exception as e:
                    data = future_to_data[future]
                    logging.error(f"Failed processing page {data[1]}: {e}")
                    results.append(f"=== PAGE {data[1]} (ERROR) ===\n")
                    if HAS_TQDM:
                        progress_bar.update(1)
        
        if HAS_TQDM:
            progress_bar.close()
        
        # Sort results by page number
        results.sort(key=lambda x: int(re.search(r'PAGE (\d+)', x).group(1)))
        return results


# Enhanced Text Cleaner
class TextCleaner:
    """Enhanced text cleaning with better pattern recognition."""

    # Patterns that might indicate prompt injection attempts
    SECURITY_PATTERNS = [
        r'(?i)ignore\s+(?:previous\s+)?instructions?',
        r'(?i)system\s+prompt\s*:',
        r'(?i)ai[\s_-]?instructions?',
        r'(?i)override\s+(?:previous\s+)?rules?',
        r'(?i)forget\s+(?:previous\s+)?instructions?',
        r'(?i)new\s+instructions?',
        r'variable\s*=\s*["\'].*["\']',
        r'set\s+\w+\s*=\s*["\'].*["\']'
    ]
    
    # Patterns for better text cleaning
    CLEANING_PATTERNS = [
        (r'=== PAGE \d+ ===\n?', ''),
        (r'--- ENHANCED ---\n?', ''),
        (r'=== PAGE \d+ \([^)]+\) ===\n?', ''),
        (r'\s+', ' '),  # Normalize whitespace
        (r'\n\s*\n+', '\n\n'),  # Remove excessive line breaks
        (r'[^\x00-\x7F]+', ''),  # Remove non-ASCII characters
    ]

    def clean_security_risks(self, text: str) -> str:
        """Remove potential security risks and prompt injection attempts."""
        original_length = len(text)
        
        for pattern in self.SECURITY_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
        
        cleaned_length = len(text)
        if cleaned_length < original_length:
            logging.warning(f"Removed {original_length - cleaned_length} characters of potentially risky content")
        
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization."""
        for pattern, replacement in self.CLEANING_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        return text.strip()

    def extract_structured_content(self, text: str) -> Dict[str, List[str]]:
        """Extract different types of content from text."""
        content = {
            'code_blocks': [],
            'requirements': [],
            'examples': [],
            'class_names': [],
            'method_signatures': []
        }
        
        # Extract code blocks
        code_patterns = [
            r'```java(.*?)```',
            r'```(.*?)```',
            r'public\s+class\s+\w+\s*\{[^}]*\}',
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            content['code_blocks'].extend(matches)
        
        # Extract requirements (lines starting with numbers, bullets, etc.)
        req_pattern = r'^(?:\d+[\.)]\s*|[-*]\s*|[a-zA-Z][\.)]\s*)(.*?)$'
        content['requirements'] = re.findall(req_pattern, text, re.MULTILINE)
        
        # Extract class names
        class_pattern = r'class\s+(\w+)'
        content['class_names'] = re.findall(class_pattern, text, re.IGNORECASE)
        
        # Extract method signatures
        method_pattern = r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)'
        content['method_signatures'] = re.findall(method_pattern, text)
        
        return content


# Enhanced AI Code Generator
class AICodeGenerator:
    """Enhanced AI code generator with better error handling and retry logic."""

    def __init__(self, api_keys: List[str], retry_attempts: int = 3, retry_delay: float = 2.0):
        if not api_keys or not any(api_keys):
            raise APIError("No valid API keys provided")
        
        self.api_keys = [key.strip() for key in api_keys if key and key.strip()]
        self.current_key_index = 0
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        try:
            import google.generativeai as genai
            self.genai = genai
        except ImportError:
            raise APIError("google-generativeai package not installed. Install with 'pip install google-generativeai'")

    def _rotate_api_key(self) -> str:
        """Get next API key in rotation."""
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key

    def _create_enhanced_prompt(self, assignment_text: str, concepts: str, strategy: str = "standard") -> str:
        """Create enhanced prompt based on strategy."""
        
        base_prompt = f"""You are an expert Java programmer creating educational code.

STRICT CONSTRAINTS:
1. Use ONLY these Java concepts: {concepts}
2. Generate COMPLETE, COMPILABLE Java code
3. Include proper class structure with main method
4. Add meaningful variable names and comments
5. Handle basic error cases
6. Follow Java naming conventions

ASSIGNMENT:
{assignment_text}

OUTPUT REQUIREMENTS:
- Start with complete Java class
- Include all necessary imports
- Add brief comments explaining key logic
- Ensure code compiles and runs
- Handle edge cases appropriately
"""

        strategy_additions = {
            "detailed": "\nSTRATEGY: Provide detailed implementation with extensive comments and error handling.",
            "simplified": "\nSTRATEGY: Create minimal but complete solution focusing on core requirements.",
            "robust": "\nSTRATEGY: Include comprehensive error handling and input validation.",
            "educational": "\nSTRATEGY: Add educational comments explaining each step and concept used."
        }
        
        return base_prompt + strategy_additions.get(strategy, "")

    def _make_api_call(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Make API call with proper error handling."""
        try:
            # Configure API
            self.genai.configure(api_key=self._rotate_api_key())
            
            # Create model instance
            model_instance = self.genai.GenerativeModel(model)
            
            # Generate content
            response = model_instance.generate_content(
                prompt,
                generation_config=self.genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    candidate_count=1
                )
            )
            
            # Extract response text
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    return candidate.content.parts[0].text.strip()
                    
            raise APIError("No valid response content received")
            
        except Exception as e:
            raise APIError(f"API call failed: {str(e)}")

    def generate_code(
        self,
        assignment_text: str,
        concepts: str,
        model: str = "gemini-1.5-pro",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        strategy: str = "standard"
    ) -> str:
        """Generate code with retry logic and fallback strategies."""
        
        prompt = self._create_enhanced_prompt(assignment_text, concepts, strategy)
        
        last_exception = None
        
        # Try each API key
        for key_attempt in range(len(self.api_keys)):
            # Retry with current key
            for attempt in range(self.retry_attempts):
                try:
                    result = self._make_api_call(prompt, model, temperature, max_tokens)
                    if result and len(result.strip()) > 50:  # Ensure meaningful response
                        return result
                    else:
                        raise APIError("Response too short or empty")
                        
                except APIError as e:
                    last_exception = e
                    if attempt < self.retry_attempts - 1:
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(self.retry_delay)
                    continue
        
        raise APIError(f"All API attempts failed. Last error: {last_exception}")

    def generate_with_fallback(
        self,
        assignment_text: str,
        concepts: str,
        model: str = "gemini-1.5-pro",
        max_tokens: int = 4000
    ) -> str:
        """Try multiple strategies if primary generation fails."""
        
        strategies = [
            ("standard", 0.1),
            ("detailed", 0.1), 
            ("simplified", 0.0),
            ("robust", 0.2),
            ("educational", 0.1)
        ]
        
        for strategy, temp in strategies:
            try:
                logging.info(f"Trying generation strategy: {strategy}")
                return self.generate_code(
                    assignment_text, concepts, model, temp, max_tokens, strategy
                )
            except Exception as e:
                logging.warning(f"Strategy '{strategy}' failed: {e}")
                continue
        
        raise APIError("All generation strategies failed")


# Enhanced Code Post-Processor
class CodePostProcessor:
    """Enhanced code post-processing with better validation."""

    JAVA_KEYWORDS = {
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch',
        'char', 'class', 'const', 'continue', 'default', 'do', 'double',
        'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
        'goto', 'if', 'implements', 'import', 'instanceof', 'int',
        'interface', 'long', 'native', 'new', 'package', 'private',
        'protected', 'public', 'return', 'short', 'static', 'strictfp',
        'super', 'switch', 'synchronized', 'this', 'throw', 'throws',
        'transient', 'try', 'void', 'volatile', 'while'
    }

    def extract_java_code(self, text: str) -> str:
        """Extract Java code from mixed content."""
        # Look for code blocks first
        code_block_patterns = [
            r'```java\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # Look for class definitions
        class_pattern = r'((?:public\s+)?class\s+\w+.*?\{.*?\n\})'
        class_matches = re.findall(class_pattern, text, re.DOTALL)
        if class_matches:
            return class_matches[0].strip()
        
        # Return cleaned text if no specific patterns found
        return self._clean_non_code_content(text)

    def _clean_non_code_content(self, text: str) -> str:
        """Remove obvious non-code content."""
        # Remove common non-code phrases
        non_code_patterns = [
            r'(?i)here\'s.*?solution.*?:?\s*',
            r'(?i)the\s+code\s+(?:is|would\s+be).*?:?\s*',
            r'(?i)solution.*?:?\s*',
            r'(?i)answer.*?:?\s*',
        ]
        
        for pattern in non_code_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        return text.strip()

    def sanitize_code(self, code: str) -> str:
        """Enhanced code sanitization."""
        # Extract just the Java code
        code = self.extract_java_code(code)
        
        # Fix common formatting issues
        code = self._fix_indentation(code)
        code = self._fix_braces(code)
        code = self._ensure_proper_spacing(code)
        
        # Add header comment
        header = "// AUTO-GENERATED CODE - REVIEW BEFORE USE\n// Generated from PDF assignment\n\n"
        
        return header + code.strip()

    def _fix_indentation(self, code: str) -> str:
        """Fix indentation to use consistent spacing."""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Convert tabs to spaces and normalize indentation
            line = line.expandtabs(4)
            # Remove trailing whitespace
            line = line.rstrip()
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _fix_braces(self, code: str) -> str:
        """Ensure proper brace formatting."""
        # Fix opening braces
        code = re.sub(r'\s*\{\s*', ' {\n', code)
        # Fix closing braces
        code = re.sub(r'\s*\}\s*', '\n}\n', code)
        return code

    def _ensure_proper_spacing(self, code: str) -> str:
        """Ensure proper spacing around operators and keywords."""
        # Add space around operators
        operators = ['=', '+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=']
        for op in operators:
            code = re.sub(f'\\s*{re.escape(op)}\\s*', f' {op} ', code)
        
        # Remove excessive blank lines
        code = re.sub(r'\n\s*\n\s*\n+', '\n\n', code)
        
        return code

    def extract_main_class(self, code: str) -> Optional[str]:
        """Extract main class name with better pattern matching."""
        patterns = [
            r'public\s+class\s+(\w+)',
            r'class\s+(\w+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, code)
            if match:
                class_name = match.group(1)
                # Ensure it's a valid Java identifier
                if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', class_name) and class_name not in self.JAVA_KEYWORDS:
                    return class_name
        
        return None

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Enhanced syntax validation."""
        issues = []
        
        # Check balanced delimiters
        delimiters = [
            ('{', '}', 'braces'),
            ('(', ')', 'parentheses'), 
            ('[', ']', 'brackets')
        ]
        
        for open_char, close_char, name in delimiters:
            opens = code.count(open_char)
            closes = code.count(close_char)
            if opens != closes:
                issues.append(f"Unbalanced {name}: {opens} opens vs {closes} closes")
        
        # Check for required elements
        required_checks = [
            (r'class\s+\w+', "Missing class declaration"),
            (r'public\s+static\s+void\s+main', "Missing main method"),
            (r'\bSystem\.out\.print', "No output statements found")
        ]
        
        for pattern, message in required_checks:
            if not re.search(pattern, code):
                issues.append(message)
        
        # Check for common syntax errors
        syntax_checks = [
            (r';\s*\n\s*}', "Semicolon before closing brace"),
            (r'if\s*\([^)]*\)\s*{[^}]*}[^}]*else', "Malformed if-else statement"),
        ]
        
        for pattern, message in syntax_checks:
            if re.search(pattern, code):
                issues.append(message)
        
        return len(issues) == 0, issues

    def analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Enhanced code quality analysis."""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        metrics = {
            'total_lines': len(code.split('\n')),
            'code_lines': len(lines),
            'classes': len(re.findall(r'class\s+\w+', code)),
            'methods': len(re.findall(r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\([^)]*\)', code)),
            'complexity': self._calculate_complexity(code),
            'comments': code.count('//') + code.count('/*'),
            'imports': len(re.findall(r'import\s+[\w.]+;', code)),
            'variables': len(set(re.findall(r'(?:int|double|String|boolean|float|char|long)\s+(\w+)', code))),
        }
        
        # Calculate quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics, code)
        
        return metrics

    def _calculate_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity_keywords = ['if', 'for', 'while', 'switch', 'catch', '&&', '||', '?']
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            complexity += code.count(keyword)
        
        return complexity

    def _calculate_quality_score(self, metrics: Dict[str, Any], code: str) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Deduct for various issues
        if metrics['comments'] == 0:
            score -= 10  # No comments
        if metrics['methods'] == 0:
            score -= 20  # No methods
        if metrics['complexity'] > 10:
            score -= (metrics['complexity'] - 10) * 2  # High complexity
        if metrics['code_lines'] < 10:
            score -= 15  # Too simple
        elif metrics['code_lines'] > 100:
            score -= 10  # Too complex
        
        return max(0.0, min(100.0, score))


# Enhanced Java Compiler
class JavaCompiler:
    """Enhanced Java compiler with better resource management and security."""

    def __init__(self, work_dir: Path = Path(".")):
        self.work_dir = work_dir
        self.temp_dir = None

    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="java_compile_"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp directory: {e}")

    def compile_and_run(
        self,
        code: str,
        class_name: str,
        timeout: int = 30
    ) -> Tuple[bool, str, str, Dict[str, Any]]:
        """Compile and execute Java code with enhanced feedback."""
        
        java_file = self.temp_dir / f"{class_name}.java"
        
        try:
            # Write code to file
            java_file.write_text(code, encoding='utf-8')
            
            # Compilation step
            compile_start = time.time()
            compile_proc = subprocess.run(
                ["javac", "-cp", str(self.temp_dir), str(java_file)],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            compile_time = time.time() - compile_start
            
            if compile_proc.returncode != 0:
                return False, "", compile_proc.stderr, {
                    'compile_time': compile_time,
                    'compilation_success': False
                }
            
            # Execution step
            run_start = time.time()
            run_proc = subprocess.run(
                ["java", "-cp", str(self.temp_dir), class_name],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            run_time = time.time() - run_start
            
            execution_metrics = {
                'compile_time': compile_time,
                'run_time': run_time,
                'compilation_success': True,
                'execution_success': run_proc.returncode == 0,
                'exit_code': run_proc.returncode
            }
            
            return (run_proc.returncode == 0, run_proc.stdout, run_proc.stderr, execution_metrics)
            
        except subprocess.TimeoutExpired:
            return False, "", f"Operation timed out after {timeout}s", {
                'compilation_success': False,
                'timeout': True
            }
        except Exception as e:
            return False, "", f"Unexpected error: {str(e)}", {
                'compilation_success': False,
                'error': str(e)
            }


# Enhanced Output Manager
class OutputManager:
    """Enhanced output manager with multiple format support."""

    def save_outputs(
        self,
        code: str,
        output_path: Path,
        formats: List[str],
        quality_metrics: Optional[Dict] = None,
        execution_results: Optional[Dict] = None
    ):
        """Save code in multiple formats with enhanced metadata."""
        base_path = output_path.with_suffix('')
        
        for fmt in formats:
            try:
                if fmt == 'java':
                    self._save_java(output_path, code)
                elif fmt == 'txt':
                    self._save_txt(base_path.with_suffix('.txt'), code)
                elif fmt == 'html':
                    self._save_html(base_path.with_suffix('.html'), code, quality_metrics, execution_results)
                elif fmt == 'json':
                    self._save_json(base_path.with_suffix('.json'), code, quality_metrics, execution_results)
                elif fmt == 'markdown':
                    self._save_markdown(base_path.with_suffix('.md'), code, quality_metrics, execution_results)
                    
            except Exception as e:
                logging.error(f"Failed to save {fmt} format: {e}")

    def _save_java(self, path: Path, code: str):
        """Save Java code file."""
        path.write_text(code, encoding='utf-8')
        logging.info(f"Java code saved to {path}")

    def _save_txt(self, path: Path, code: str):
        """Save plain text version."""
        path.write_text(code, encoding='utf-8')
        logging.info(f"Text version saved to {path}")

    def _save_html(self, path: Path, code: str, quality_metrics: Dict = None, execution_results: Dict = None):
        """Save HTML version with syntax highlighting."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Java Code</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .code {{ background: #f8f8f8; border: 1px solid #ddd; border-radius: 5px; padding: 20px; font-family: 'Courier New', monospace; white-space: pre-wrap; overflow-x: auto; }}
        .metrics {{ background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ margin: 5px 0; }}
        .success {{ color: #28a745; }}
        .error {{ color: #dc3545; }}
        .warning {{ color: #ffc107; }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Generated Java Code</h1>
        <div class="code">{self._escape_html(code)}</div>"""

        if quality_metrics:
            html_content += self._generate_metrics_html(quality_metrics)
            
        if execution_results:
            html_content += self._generate_execution_html(execution_results)
            
        html_content += """
    </div>
</body>
</html>"""
        
        path.write_text(html_content, encoding='utf-8')
        logging.info(f"HTML version saved to {path}")

    def _save_json(self, path: Path, code: str, quality_metrics: Dict = None, execution_results: Dict = None):
        """Save JSON version with all metadata."""
        output_data = {
            "timestamp": time.time(),
            "code": code,
            "quality_metrics": quality_metrics or {},
            "execution_results": execution_results or {}
        }
        
        path.write_text(json.dumps(output_data, indent=2, default=str), encoding='utf-8')
        logging.info(f"JSON version saved to {path}")

    def _save_markdown(self, path: Path, code: str, quality_metrics: Dict = None, execution_results: Dict = None):
        """Save Markdown version."""
        md_content = f"""# Generated Java Code

## Source Code

```java
{code}
```
"""
        
        if quality_metrics:
            md_content += "\n## Code Quality Metrics\n\n"
            for key, value in quality_metrics.items():
                md_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        if execution_results:
            md_content += "\n## Execution Results\n\n"
            for key, value in execution_results.items():
                md_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        path.write_text(md_content, encoding='utf-8')
        logging.info(f"Markdown version saved to {path}")

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _generate_metrics_html(self, metrics: Dict) -> str:
        """Generate HTML for quality metrics."""
        html = "<h2>Code Quality Metrics</h2><div class='metrics'>"
        
        for key, value in metrics.items():
            css_class = ""
            if key == 'quality_score':
                if value >= 80:
                    css_class = "success"
                elif value >= 60:
                    css_class = "warning"
                else:
                    css_class = "error"
            
            html += f"<div class='metric {css_class}'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
        
        html += "</div>"
        return html

    def _generate_execution_html(self, results: Dict) -> str:
        """Generate HTML for execution results."""
        html = "<h2>Execution Results</h2><div class='metrics'>"
        
        for key, value in results.items():
            css_class = ""
            if key in ['compilation_success', 'execution_success'] and isinstance(value, bool):
                css_class = "success" if value else "error"
            
            html += f"<div class='metric {css_class}'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>"
        
        html += "</div>"
        return html


# Enhanced Main Processor
class PDFToJavaProcessor:
    """Enhanced main processor with better error handling and reporting."""

    def __init__(self, config: ConfigManager, api_keys: List[str], verbose: bool = False):
        self.config = config
        self.verbose = verbose
        
        # Initialize components
        self.ocr = OCRProcessor(
            contrast_factor=config["contrast_factor"],
            dpi=config["dpi"]
        )
        self.cleaner = TextCleaner()
        self.ai = AICodeGenerator(
            api_keys=api_keys,
            retry_attempts=config["retry_attempts"],
            retry_delay=config["retry_delay"]
        )
        self.post = CodePostProcessor()
        self.output = OutputManager()

    def process_pdf(
        self,
        pdf_path: Path,
        concepts_path: Path,
        output_path: Path,
    ) -> bool:
        """Enhanced PDF processing pipeline."""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(pdf_path, concepts_path)
            
            # Check PDF page count
            page_info = pdfinfo_from_path(pdf_path)
            page_count = page_info.get('Pages', 0)
            
            if page_count > self.config["max_pages"]:
                logging.warning(f"PDF has {page_count} pages, limiting to {self.config['max_pages']}")
            
            # OCR Processing
            extracted_text = self._extract_text_from_pdf(pdf_path)
            
            # Text Processing
            processed_text = self._process_extracted_text(extracted_text)
            
            # Load concepts
            concepts = self._load_concepts(concepts_path)
            
            # AI Code Generation
            generated_code = self._generate_code_with_ai(processed_text, concepts)
            
            # Post-processing
            final_code, class_name = self._post_process_code(generated_code, output_path)
            
            # Validation and Quality Analysis
            quality_metrics = self._analyze_code_quality(final_code)
            
            # Compilation and Execution
            execution_results = self._test_code_execution(final_code, class_name)
            
            # Save outputs
            self._save_all_outputs(final_code, output_path, quality_metrics, execution_results)
            
            # Final reporting
            total_time = time.time() - start_time
            self._generate_final_report(quality_metrics, execution_results, total_time)
            
            return execution_results.get('compilation_success', False)
            
        except Exception as e:
            logging.error(f"Processing failed: {e}")
            if self.verbose:
                import traceback
                logging.debug(traceback.format_exc())
            return False

    def _validate_inputs(self, pdf_path: Path, concepts_path: Path):
        """Validate input files."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if not concepts_path.exists():
            raise FileNotFoundError(f"Concepts file not found: {concepts_path}")
        if pdf_path.stat().st_size == 0:
            raise ValueError("PDF file is empty")
        if concepts_path.stat().st_size == 0:
            raise ValueError("Concepts file is empty")

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF with caching."""
        # Try cache first
        if self.config["cache_ocr"]:
            cached_results = self.ocr.load_cached_results(pdf_path)
            if cached_results:
                logging.info("Using cached OCR results")
                return "\n".join(cached_results)
        
        # Convert PDF to images
        logging.info("Converting PDF to images...")
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.config["dpi"],
                thread_count=self.config["workers"],
                first_page=1,
                last_page=self.config["max_pages"]
            )
        except Exception as e:
            raise PDFProcessorError(f"Failed to convert PDF: {e}")
        
        if not images:
            raise PDFProcessorError("No pages extracted from PDF")
        
        # Process with OCR
        logging.info(f"Processing {len(images)} pages with OCR...")
        ocr_results = self.ocr.process_with_progress(images)
        
        # Cache results
        if self.config["cache_ocr"]:
            self.ocr.save_cached_results(pdf_path, ocr_results)
        
        return "\n".join(ocr_results)

    def _process_extracted_text(self, raw_text: str) -> str:
        """Process and clean extracted text."""
        # Security cleaning
        clean_text = self.cleaner.clean_security_risks(raw_text)
        
        # Normalization
        normalized_text = self.cleaner.normalize_text(clean_text)
        
        # Extract structured content for better understanding
        structured_content = self.cleaner.extract_structured_content(normalized_text)
        
        if self.verbose:
            logging.debug(f"Extracted {len(normalized_text)} characters")
            logging.debug(f"Found {len(structured_content['requirements'])} requirements")
            logging.debug(f"Found {len(structured_content['code_blocks'])} code blocks")
        
        return normalized_text

    def _load_concepts(self, concepts_path: Path) -> str:
        """Load and validate concepts file."""
        try:
            concepts = concepts_path.read_text(encoding='utf-8').strip()
            if not concepts:
                raise ValueError("Concepts file is empty")
            return concepts
        except Exception as e:
            raise PDFProcessorError(f"Failed to load concepts: {e}")

    def _generate_code_with_ai(self, text: str, concepts: str) -> str:
        """Generate code using AI with fallback strategies."""
        logging.info("Generating Java code with AI...")
        
        try:
            return self.ai.generate_code(
                text,
                concepts,
                model=self.config["model"],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"]
            )
        except Exception as e:
            logging.warning(f"Primary generation failed: {e}")
            logging.info("Trying fallback strategies...")
            return self.ai.generate_with_fallback(
                text,
                concepts,
                model=self.config["model"],
                max_tokens=self.config["max_tokens"]
            )

    def _post_process_code(self, raw_code: str, output_path: Path) -> Tuple[str, str]:
        """Post-process generated code."""
        # Clean and format code
        cleaned_code = self.post.sanitize_code(raw_code)
        
        # Extract class name
        class_name = self.post.extract_main_class(cleaned_code)
        if not class_name:
            class_name = output_path.stem
            # Wrap in class if needed
            if not re.search(r'class\s+\w+', cleaned_code):
                cleaned_code = f"""public class {class_name} {{
    public static void main(String[] args) {{
        {cleaned_code}
    }}
}}"""
                logging.info(f"Added class wrapper: {class_name}")
        
        return cleaned_code, class_name

    def _analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality and syntax."""
        # Basic quality metrics
        quality_metrics = self.post.analyze_code_quality(code)
        
        # Syntax validation
        if self.config["validate"]:
            valid, issues = self.post.validate_syntax(code)
            quality_metrics['syntax_valid'] = valid
            quality_metrics['syntax_issues'] = issues
            
            if not valid:
                logging.warning(f"Syntax issues found: {', '.join(issues)}")
        
        return quality_metrics

    def _test_code_execution(self, code: str, class_name: str) -> Dict[str, Any]:
        """Test code compilation and execution."""
        execution_results = {}
        
        if "java" in self.config["output_formats"]:
            try:
                with JavaCompiler() as compiler:
                    success, stdout, stderr, metrics = compiler.compile_and_run(
                        code, class_name, timeout=self.config["timeout"]
                    )
                    
                    execution_results.update(metrics)
                    execution_results['stdout'] = stdout[:500] if stdout else ""  # Limit output
                    execution_results['stderr'] = stderr[:500] if stderr else ""
                    
                    if success:
                        logging.info("✅ Code compiled and executed successfully")
                        if stdout:
                            logging.info(f"Program output: {stdout[:100]}{'...' if len(stdout) > 100 else ''}")
                    else:
                        logging.error(f"❌ Execution failed: {stderr}")
                        
            except Exception as e:
                logging.error(f"Compilation test failed: {e}")
                execution_results['error'] = str(e)
        
        return execution_results

    def _save_all_outputs(self, code: str, output_path: Path, quality_metrics: Dict, execution_results: Dict):
        """Save outputs in all requested formats."""
        self.output.save_outputs(
            code=code,
            output_path=output_path,
            formats=self.config["output_formats"],
            quality_metrics=quality_metrics,
            execution_results=execution_results
        )

    def _generate_final_report(self, quality_metrics: Dict, execution_results: Dict, total_time: float):
        """Generate final processing report."""
        logging.info("=" * 60)
        logging.info("PROCESSING SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total processing time: {total_time:.2f}s")
        
        if quality_metrics:
            logging.info(f"Code quality score: {quality_metrics.get('quality_score', 'N/A')}/100")
            logging.info(f"Lines of code: {quality_metrics.get('code_lines', 'N/A')}")
            logging.info(f"Complexity score: {quality_metrics.get('complexity', 'N/A')}")
        
        if execution_results:
            compilation = "✅ SUCCESS" if execution_results.get('compilation_success') else "❌ FAILED"
            logging.info(f"Compilation: {compilation}")
            
            if execution_results.get('compile_time'):
                logging.info(f"Compile time: {execution_results['compile_time']:.3f}s")
        
        logging.info("=" * 60)


# Enhanced Dependency Checker
def check_dependencies():
    """Enhanced dependency checking with detailed error messages."""
    requirements = {
        "tesseract": {
            "cmd": ["tesseract", "--version"],
            "install": "Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract"
        },
        "javac": {
            "cmd": ["javac", "-version"],
            "install": "Install Java JDK: https://adoptopenjdk.net/"
        },
        "pdftoppm": {
            "cmd": ["pdftoppm", "-v"],
            "install": "Install poppler-utils (Linux: apt install poppler-utils, macOS: brew install poppler)"
        }
    }

    missing = []
    for tool, info in requirements.items():
        try:
            result = subprocess.run(
                info["cmd"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL, 
                check=True,
                timeout=10
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            missing.append((tool, info["install"]))

    if missing:
        error_msg = "Missing dependencies:\n"
        for tool, install_info in missing:
            error_msg += f"  - {tool}: {install_info}\n"
        raise PDFProcessorError(error_msg)

    logging.info("✅ All dependencies are available")


def get_api_keys() -> List[str]:
    """Enhanced API key loading with validation."""
    # Try multiple environment variable names
    env_vars = ["GEMINI_API_KEYS", "GOOGLE_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    
    for env_var in env_vars:
        keys_str = os.getenv(env_var, "")
        if keys_str:
            keys = [key.strip() for key in keys_str.split(",") if key.strip()]
            if keys:
                logging.info(f"Found {len(keys)} API key(s) from {env_var}")
                return keys
    
    # Try loading from file
    key_files = [".api_keys", "api_keys.txt", "keys.txt"]
    for key_file in key_files:
        key_path = Path(key_file)
        if key_path.exists():
            try:
                content = key_path.read_text().strip()
                keys = [key.strip() for key in content.split('\n') if key.strip() and not key.startswith('#')]
                if keys:
                    logging.info(f"Found {len(keys)} API key(s) from {key_file}")
                    return keys
            except Exception as e:
                logging.warning(f"Failed to read {key_file}: {e}")
    
    return []


# Enhanced Main Function
def main():
    """Enhanced main function with better argument handling."""
    parser = argparse.ArgumentParser(
        description="Enhanced PDF to Java Code Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdf assignment.pdf --concepts java_concepts.txt
  %(prog)s --pdf hw1.pdf --concepts concepts.txt --output Solution.java --verbose
  %(prog)s --pdf test.pdf --concepts allowed.txt --formats java html json
        """
    )
    
    # Required arguments
    parser.add_argument("--pdf", type=Path, required=True,
                       help="Input PDF file path")
    parser.add_argument("--concepts", type=Path, required=True,
                       help="File containing allowed Java concepts/keywords")
    
    # Optional arguments
    parser.add_argument("--output", type=Path, default=Path("Solution.java"),
                       help="Output Java file path")
    parser.add_argument("--config", type=Path,
                       help="JSON configuration file path")
    
    # Processing options
    parser.add_argument("--dpi", type=int, metavar="N",
                       help="Image resolution for OCR (150-600)")
    parser.add_argument("--model", choices=ConfigManager.VALID_MODELS,
                       help="Gemini model to use")
    parser.add_argument("--temperature", type=float, metavar="N",
                       help="AI creativity level (0.0-2.0)")
    parser.add_argument("--max-tokens", type=int, metavar="N",
                       help="Maximum tokens for AI response")
    parser.add_argument("--workers", type=int, metavar="N",
                       help="Number of OCR processing threads")
    parser.add_argument("--timeout", type=int, metavar="N",
                       help="Timeout for compilation/execution (seconds)")
    
    # Feature flags
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip syntax validation")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable OCR result caching")
    parser.add_argument("--formats", nargs="+", 
                       choices=ConfigManager.VALID_FORMATS,
                       help="Output formats to generate")
    
    # Logging
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable detailed debug output")
    parser.add_argument("--log-file", type=Path,
                       help="Save logs to specified file")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_handlers = [logging.StreamHandler()]
    
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )

    try:
        # Print banner
        logging.info("🚀 Enhanced PDF to Java Code Generator")
        logging.info("=" * 50)
        
        # Check dependencies
        logging.info("Checking system dependencies...")
        check_dependencies()
        
        # Get API keys
        api_keys = get_api_keys()
        if not api_keys:
            raise PDFProcessorError(
                "No API keys found. Set one of these environment variables:\n"
                "  - GEMINI_API_KEYS (comma-separated for multiple keys)\n"
                "  - GEMINI_API_KEY (single key)\n"
                "Or create a file named '.api_keys' with one key per line"
            )
        
        # Setup configuration
        config = ConfigManager()
        
        # Load config file if specified
        if args.config:
            if config.load_file(args.config):
                logging.info(f"Configuration loaded from {args.config}")
            else:
                logging.warning(f"Failed to load config file: {args.config}")
        
        # Apply command line arguments
        config.update_from_args(args)
        
        if args.verbose:
            logging.debug(f"Final configuration: {json.dumps(config.config, indent=2)}")
        
        # Initialize and run processor
        processor = PDFToJavaProcessor(
            config=config,
            api_keys=api_keys,
            verbose=args.verbose
        )
        
        success = processor.process_pdf(
            pdf_path=args.pdf,
            concepts_path=args.concepts,
            output_path=args.output
        )
        
        if success:
            logging.info("🎉 Processing completed successfully!")
            sys.exit(0)
        else:
            logging.error("💥 Processing failed!")
            sys.exit(1)
            
    except PDFProcessorError as e:
        logging.error(f"❌ Configuration/Processing error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("⚠️  Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"💀 Unexpected error: {e}")
        if args.verbose:
            import traceback
            logging.debug(traceback.format_exc())
        sys.exit(2)


if __name__ == "__main__":
    main()
