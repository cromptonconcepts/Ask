"""
Multimodal Image Extraction and Vision Description

Extracts images/diagrams from PDFs and generates descriptions using vision models.
Integrates with Ollama's multimodal models (llava, qwen2.5vl, etc.) for image understanding.
"""

import os
import io
import base64
from pathlib import Path
from typing import List, Tuple, Optional, Any, cast
import logging

fitz = None
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

Image = None
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

ollama = None
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

logger = logging.getLogger(__name__)


def _model_name_from_entry(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get('name') or entry.get('model') or '')
    return str(getattr(entry, 'name', '') or getattr(entry, 'model', '') or '')

# Vision model configuration - models that support image input
SUPPORTED_VISION_MODELS = {
    "llava": {
        "name": "llava",
        "description": "Llava 1.6 - Good for diagrams and charts",
        "context_size": 4096,
        "parameter_size": "34b"
    },
    "qwen2.5vl": {
        "name": "qwen2.5vl",
        "description": "Qwen2.5-VL - Excellent for technical diagrams and document understanding",
        "context_size": 4096,
        "parameter_size": "3b/7b/32b/72b"
    },
    "bakllava": {
        "name": "bakllava",
        "description": "BakLLaVA - Faster alternative to Llava",
        "context_size": 2048,
        "parameter_size": "13b"
    }
}

DEFAULT_VISION_MODEL = "qwen2.5vl"  # Recommended for traffic diagrams and technical docs

PILImageType = Any


class MultimodalImageExtractor:
    """Extract images from PDFs and generate vision descriptions."""
    
    def __init__(
        self,
        vision_model: str = DEFAULT_VISION_MODEL,
        base_url: str | None = None,
        fallback_vision_model: str | None = None,
        fallback_base_url: str | None = None,
        request_timeout_seconds: int = 90,
    ):
        """
        Initialize the image extractor.
        
        Args:
            vision_model: Vision model to use (qwen2.5vl, llava, bakllava)
            base_url: Ollama server base URL
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")
        if not HAS_PIL:
            raise ImportError("Pillow not installed. Install with: pip install Pillow")
        if not HAS_OLLAMA:
            raise ImportError("ollama not installed. Install with: pip install ollama")

        ollama_module = cast(Any, ollama)
        
        self.vision_model = vision_model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).strip() or "http://localhost:11434"
        self.request_timeout_seconds = max(15, int(request_timeout_seconds))
        self.client = ollama_module.Client(host=self.base_url, timeout=self.request_timeout_seconds)
        self.fallback_vision_model = (fallback_vision_model or "").strip() or None
        self.fallback_base_url = (fallback_base_url or "").strip() or None
        self.fallback_client = None
        if self.fallback_vision_model and self.fallback_base_url:
            same_backend = self.fallback_vision_model == self.vision_model and self.fallback_base_url.rstrip('/') == self.base_url.rstrip('/')
            if not same_backend:
            self.fallback_client = ollama_module.Client(host=self.fallback_base_url, timeout=self.request_timeout_seconds)
        
        # Verify model is available
        self._verify_model_available()
    
    def _verify_model_available(self):
        """Check if the vision model is available in Ollama."""
        try:
            # Try to get model info
            models = self.client.list()
            model_entries = getattr(models, 'models', None)
            if model_entries is None and isinstance(models, dict):
                model_entries = models.get('models', [])
            if model_entries is None:
                model_entries = []

            available_models = []
            for model_entry in model_entries:
                model_name = _model_name_from_entry(model_entry)
                if model_name:
                    available_models.append(model_name.split(':')[0])
            
            if self.vision_model not in available_models:
                print(f"Warning: Vision model '{self.vision_model}' not found locally.")
                print(f"Available models: {', '.join(available_models)}")
                print("To add the vision model, run:")
                print(f"ollama pull {self.vision_model}")
                raise ValueError(f"Vision model '{self.vision_model}' not available")
            
            print(f"Vision model '{self.vision_model}' available")
        except Exception as e:
            if "connection" in str(e).lower() or "refused" in str(e).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running: ollama serve"
                )
            raise
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[int, PILImageType]]:
        """
        Extract all images from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of (page_number, PIL Image) tuples
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        images: List[Tuple[int, PILImageType]] = []
        try:
            fitz_module = cast(Any, fitz)
            image_module = cast(Any, Image)
            pdf = fitz_module.open(pdf_path)
            
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                # Extract images from page
                image_list = page.get_images()
                
                for img_index, image_info in enumerate(image_list):
                    try:
                        # Get image data
                        xref = image_info[0]
                        pix = fitz_module.Pixmap(pdf, xref)
                        
                        # Filter out small images (likely artifacts)
                        if pix.width < 50 or pix.height < 50:
                            pix = None
                            continue
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                        else:  # CMYK
                            pix = fitz_module.Pixmap(fitz_module.csRGB, pix)
                            img_data = pix.tobytes("ppm")
                        
                        img = image_module.open(io.BytesIO(img_data))
                        images.append((page_num + 1, img))  # 1-indexed pages
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            pdf.close()
            print(f"Extracted {len(images)} images from {pdf_path}")
            return images
            
        except Exception as e:
            raise ValueError(f"Failed to extract images from PDF '{pdf_path}': {str(e)}")
    
    def encode_image_to_base64(self, image: PILImageType) -> str:
        """
        Encode PIL Image to base64 string for Ollama API.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = io.BytesIO()
        image_module = cast(Any, Image)
        
        # Convert RGBA to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = image_module.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            rgb_image.save(buffered, format="PNG")
        else:
            image.save(buffered, format="PNG")
        
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image_description(self, image: PILImageType, context: str = "") -> str:
        """
        Generate a text description of an image using the vision model.
        
        Args:
            image: PIL Image object
            context: Optional context about the document (e.g., "Traffic management diagram")
            
        Returns:
            Text description of the image
        """
        # Encode image
        image_base64 = self.encode_image_to_base64(image)

        prompt = (
            "Analyze this image and provide a concise, detailed description. "
            "Focus on: structures, diagrams, charts, text, numbers, equipment shown. "
            "Be specific and factual."
        )

        if context:
            prompt = f"Context: {context}\n\n{prompt}"

        def _generate(client: Any, model_name: str) -> str:
            response = client.generate(
                model=model_name,
                prompt=prompt,
                images=[image_base64],
                stream=False
            )
            description = response.get('response', '').strip()
            if not description:
                logger.warning("Vision model returned empty description")
                return "[Image description could not be generated]"
            return description

        try:
            return _generate(self.client, self.vision_model)
        except Exception as e:
            if self.fallback_client and self.fallback_vision_model:
                logger.warning(
                    "Vision model call failed on %s (%s). Retrying with local fallback %s at %s.",
                    self.vision_model,
                    self.base_url,
                    self.fallback_vision_model,
                    self.fallback_base_url,
                )
                try:
                    return _generate(self.fallback_client, self.fallback_vision_model)
                except Exception as fallback_exc:
                    logger.error("Fallback vision model error: %s", fallback_exc)
                    raise fallback_exc

            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                logger.error(f"Cannot connect to Ollama for vision model. Ensure 'ollama serve' is running")
            else:
                logger.error(f"Vision model error: {str(e)}")
            raise
    
    def extract_and_describe_images(
        self, 
        pdf_path: str, 
        context: str = "",
        max_images: int | None = None,
    ) -> List[dict]:
        """
        Extract images from PDF and generate descriptions for each.
        
        Args:
            pdf_path: Path to PDF file
            context: Optional context about the document
            
        Returns:
            List of dicts with keys: page, image, description, base64
        """
        images = self.extract_images_from_pdf(pdf_path)
        if max_images is not None:
            capped = max(0, int(max_images))
            if capped == 0:
                return []
            images = images[:capped]
        results = []
        
        for page_num, image in images:
            try:
                print(f"  Describing image from page {page_num}...")
                description = self.generate_image_description(image, context)
                
                results.append({
                    'page': page_num,
                    'image': image,
                    'description': description,
                    'base64': self.encode_image_to_base64(image),
                    'size': image.size
                })
                
            except Exception as e:
                logger.error(f"Failed to describe image on page {page_num}: {e}")
                continue
        
        return results


def extract_images_with_descriptions(
    pdf_path: str,
    vision_model: str = DEFAULT_VISION_MODEL,
    base_url: str | None = None,
    fallback_vision_model: str | None = None,
    fallback_base_url: str | None = None,
    context: str = "",
    max_images: int | None = None,
    request_timeout_seconds: int = 90,
) -> List[dict]:
    """
    Convenience function to extract and describe images from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        vision_model: Vision model to use
        context: Optional document context
        
    Returns:
        List of image description dictionaries
    """
    extractor = MultimodalImageExtractor(
        vision_model=vision_model,
        base_url=base_url,
        fallback_vision_model=fallback_vision_model,
        fallback_base_url=fallback_base_url,
        request_timeout_seconds=request_timeout_seconds,
    )
    return extractor.extract_and_describe_images(pdf_path, context, max_images=max_images)


def format_image_description_for_embedding(image_data: dict) -> str:
    """
    Format image information for embedding as text.
    
    Args:
        image_data: Dict from extract_and_describe_images
        
    Returns:
        Formatted text for embedding
    """
    return (
        f"[Image from page {image_data['page']} - {image_data['size'][0]}x{image_data['size'][1]}px]\n"
        f"Description: {image_data['description']}\n"
        f"Image size: {image_data['size'][0]}x{image_data['size'][1]} pixels"
    )


# ============================================================================
# BATCH PROCESSING FOR DIRECTORY OF PDFS
# ============================================================================

def batch_extract_images_from_pdfs(
    directory: str,
    vision_model: str = DEFAULT_VISION_MODEL,
    base_url: str | None = None,
    fallback_vision_model: str | None = None,
    fallback_base_url: str | None = None,
    max_images_per_pdf: int = 10,
    recursive: bool = True
) -> dict:
    """
    Extract and describe images from all PDFs in a directory.
    
    Args:
        directory: Directory containing PDFs
        vision_model: Vision model to use
        max_images_per_pdf: Maximum images to process per PDF
        recursive: Whether to search subdirectories
        
    Returns:
        Dict mapping PDF paths to lists of image descriptions
    """
    extractor = MultimodalImageExtractor(
        vision_model=vision_model,
        base_url=base_url,
        fallback_vision_model=fallback_vision_model,
        fallback_base_url=fallback_base_url,
    )
    results = {}
    
    # Find all PDFs
    if recursive:
        pdf_paths = Path(directory).rglob("*.pdf")
    else:
        pdf_paths = Path(directory).glob("*.pdf")
    
    pdf_paths = sorted(pdf_paths)
    
    for pdf_path in pdf_paths:
        pdf_path_str = str(pdf_path)
        print(f"\nProcessing: {pdf_path_str}")
        
        try:
            images = extractor.extract_and_describe_images(
                pdf_path_str,
                context=os.path.basename(pdf_path_str)
            )
            
            # Limit to max_images_per_pdf
            if len(images) > max_images_per_pdf:
                print(f"  Limiting to {max_images_per_pdf} images (found {len(images)})")
                images = images[:max_images_per_pdf]
            
            results[pdf_path_str] = images
            print(f"  Successfully processed {len(images)} images")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path_str}: {e}")
            results[pdf_path_str] = []
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VISION_MODEL
        
        print(f"Extracting images from: {pdf_file}")
        print(f"Using vision model: {model}\n")
        
        try:
            results = extract_images_with_descriptions(pdf_file, vision_model=model)
            for i, result in enumerate(results, 1):
                print(f"\nImage {i} (Page {result['page']}):")
                print(f"  Size: {result['size'][0]}x{result['size'][1]}px")
                print(f"  Description: {result['description']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python multimodal_image_extractor.py <pdf_path> [vision_model]")
        print(f"Available models: {', '.join(SUPPORTED_VISION_MODELS.keys())}")
