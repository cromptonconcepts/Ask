"""
PDF Markdown Extractor using PyMuPDF and table-transformer for accurate, page-level data extraction.

This version has been re-engineered to provide precise, chunk-level metadata for every piece of content extracted from a PDF. Instead of treating a PDF as a single document, it processes it page by page, and block by block.

Key enhancements:
- Page-by-Page Processing: Extracts content one page at a time to ensure page numbers are always accurate.
- Block-Level Chunking: Each paragraph, table, or image is treated as a separate "chunk" (a LangChain `Document`).
- Rich Metadata: Every chunk includes:
    - `source`: The absolute path to the PDF.
    - `page`: The 0-indexed page number.
    - `block_id`: A unique identifier for the chunk within the document (e.g., `doc_abcde_page_0_block_5`).
    - `block_type`: The type of content ('text', 'table', 'image').
    - `bbox`: The bounding box coordinates `[x0, y0, x1, y1]` for the chunk on the page.
- Markdown Table Preservation: It uses `page.find_tables()` to robustly identify tables and converts them into clean markdown format.
- Unique Document ID: A unique ID is generated for each PDF file to ensure that chunks from different documents are never mixed up.

This approach enables highly accurate source referencing, allowing a front-end application to link not just to a document, but to the specific page and even highlight the exact content block from which an answer was derived.
"""

import os
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
import fitz  # PyMuPDF
import uuid

# Multimodal Support (optional)
MultimodalImageExtractor = None
format_image_description_for_embedding = None
DEFAULT_VISION_MODEL = "qwen2.5vl"
try:
    from multimodal_image_extractor import (
        MultimodalImageExtractor,
        format_image_description_for_embedding,
        DEFAULT_VISION_MODEL
    )
    HAS_MULTIMODAL = True
except ImportError:
    HAS_MULTIMODAL = False

logger = logging.getLogger(__name__)


def format_table_as_markdown(table_data: list) -> str:
    """Converts a list of lists into a markdown table."""
    if not table_data:
        return ""

    # Helper to clean up cell content
    def clean_cell(cell):
        if cell is None:
            return ""
        return str(cell).replace("\n", " ").strip()

    header = "| " + " | ".join(clean_cell(c) for c in table_data[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
    body = "\n".join(["| " + " | ".join(clean_cell(c) for c in row) + " |" for row in table_data[1:]])

    return f"{header}\n{separator}\n{body}"

def pdf_to_documents(
    pdf_path: str,
    include_images: bool = True,
    vision_model: str | None = DEFAULT_VISION_MODEL,
    vision_base_url: str | None = None,
    fallback_vision_model: str | None = None,
    fallback_vision_base_url: str | None = None,
    max_images_per_pdf: int = 5,
    vision_request_timeout_seconds: int = 90,
) -> List[Document]:
    """
    Convert a PDF file to a list of LangChain Document objects, chunked by page and block.

    Args:
        pdf_path: Path to the PDF file.
        include_images: Whether to extract images and generate descriptions.
        vision_model: The vision model to use for image descriptions.
        vision_base_url: Ollama endpoint for image descriptions.
        fallback_vision_model: Local fallback model for image descriptions.
        fallback_vision_base_url: Local fallback Ollama endpoint for image descriptions.
        max_images_per_pdf: Maximum number of images to process per PDF.

    Returns:
        A list of LangChain Document objects, where each document represents a
        text block, a table, or an image description from the PDF.
    """
    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF '{pdf_path}': {e}")
        return []

    doc_id = str(uuid.uuid4())[:8]  # Unique ID for this document
    docs = []
    abs_pdf_path = os.path.abspath(pdf_path)

    for page_num in range(len(pdf_doc)):
        try:
            page = pdf_doc.load_page(page_num)
            
            # 1. Extract tables first and get their bounding boxes
            table_finder = page.find_tables()
            tables = list(getattr(table_finder, "tables", []) or [])
            table_bboxes = [tuple(t.bbox) for t in tables]

            for i, table in enumerate(tables):
                table_data = table.extract()
                table_md = format_table_as_markdown(table_data)
                if not table_md:
                    continue

                block_id = f"doc_{doc_id}_page_{page_num}_block_{i}_table"
                metadata = {
                    "source": abs_pdf_path,
                    "doc_id": doc_id,
                    "page": page_num,
                    "block_id": block_id,
                    "block_type": "table",
                    "bbox": list(table.bbox),
                }
                docs.append(Document(page_content=table_md, metadata=metadata))

            # 2. Extract text blocks, ignoring those that are inside table bounding boxes
            text_dict = cast(dict[str, Any], page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES))
            text_blocks = cast(list[dict[str, Any]], text_dict.get("blocks", []))
            
            block_idx = len(tables) # Start block index after tables
            for block in text_blocks:
                if block.get('type') == 0 and 'lines' in block:  # It's a text block
                    bbox_data = block.get('bbox')
                    if not isinstance(bbox_data, (list, tuple)) or len(bbox_data) < 4:
                        continue
                    block_bbox = tuple(bbox_data)
                    
                    # Check if the block is inside any table's bounding box
                    is_in_table = False
                    for table_bbox in table_bboxes:
                        # Simple overlap check
                        if not (block_bbox[2] < table_bbox[0] or block_bbox[0] > table_bbox[2] or
                                block_bbox[3] < table_bbox[1] or block_bbox[1] > table_bbox[3]):
                            is_in_table = True
                            break
                    
                    if is_in_table:
                        continue
                        
                    block_text = ""
                    lines = cast(list[dict[str, Any]], block.get('lines', []))
                    for line in lines:
                        spans = cast(list[dict[str, Any]], line.get('spans', []))
                        for span in spans:
                            block_text += str(span.get('text', '')) + " "
                        block_text += "\n"
                    
                    block_text = block_text.strip()
                    if not block_text:
                        continue

                    block_id = f"doc_{doc_id}_page_{page_num}_block_{block_idx}_text"
                    metadata = {
                        "source": abs_pdf_path,
                        "doc_id": doc_id,
                        "page": page_num,
                        "block_id": block_id,
                        "block_type": "text",
                        "bbox": list(block_bbox),
                    }
                    docs.append(Document(page_content=block_text, metadata=metadata))
                    block_idx += 1

            # 3. (Optional) Extract images and generate descriptions
            if include_images:
                # This part can be integrated more deeply if descriptions need to be associated
                # with their original location on the page. For now, we use the existing logic.
                # Note: The existing multimodal extractor doesn't provide bounding boxes.
                pass


        except Exception as e:
            logger.error(f"Error processing page {page_num} of '{pdf_path}': {e}")
            continue

    pdf_doc.close()

    if include_images and HAS_MULTIMODAL and MultimodalImageExtractor and format_image_description_for_embedding:
        try:
            image_extractor = MultimodalImageExtractor(
                vision_model=vision_model or DEFAULT_VISION_MODEL,
                base_url=vision_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                fallback_vision_model=fallback_vision_model,
                fallback_base_url=fallback_vision_base_url,
                request_timeout_seconds=vision_request_timeout_seconds,
            )
            image_context = (
                "Traffic management plan or guidance scheme drawing. "
                "Focus on signage, device placement, lane closures, tapers, buffers, barriers, "
                "pedestrian or cyclist paths, notes, legends, and dimensions."
            )
            image_results = image_extractor.extract_and_describe_images(
                pdf_path,
                context=image_context,
                max_images=max_images_per_pdf,
            )
            for image_index, image_data in enumerate(image_results, start=1):
                page_number = max(int(image_data.get("page", 1)) - 1, 0)
                block_id = f"doc_{doc_id}_page_{page_number}_block_image_{image_index}"
                metadata = {
                    "source": abs_pdf_path,
                    "doc_id": doc_id,
                    "page": page_number,
                    "block_id": block_id,
                    "block_type": "image",
                    "bbox": None,
                }
                docs.append(
                    Document(
                        page_content=format_image_description_for_embedding(image_data),
                        metadata=metadata,
                    )
                )
        except Exception as e:
            logger.warning(f"Image extraction skipped for '{pdf_path}': {e}")

    return docs

def extract_pdfs_from_directory(
    directory: str,
    recursive: bool = True,
    include_images: bool = True,
    vision_model: str = DEFAULT_VISION_MODEL,
    vision_base_url: str | None = None,
    fallback_vision_model: str | None = None,
    fallback_vision_base_url: str | None = None,
    max_images_per_pdf: int = 5,
    max_workers: int = 4,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> List[Document]:
    """
    Extract all PDFs from a directory and return as LangChain Documents, chunked by content blocks.
    
    Uses parallel processing via ThreadPoolExecutor for faster extraction.
    
    Args:
        directory: Root directory containing PDFs.
        recursive: Whether to search subdirectories.
        max_workers: Number of parallel threads for PDF extraction.
        
    Returns:
        A list of all Document objects from all PDFs in the directory, chunked by block.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
    
    pdf_paths = sorted(Path(directory).rglob("*.pdf") if recursive else Path(directory).glob("*.pdf"))
    pdf_path_strs = [str(p) for p in pdf_paths]
    
    if not pdf_path_strs:
        print(f"No PDFs found in {directory}")
        if progress_callback:
            progress_callback({
                "event": "complete",
                "total_pdfs": 0,
                "processed_pdfs": 0,
                "remaining_pdfs": 0,
            })
        return []
    
    all_documents = []
    total_pdfs = len(pdf_path_strs)
    if progress_callback:
        progress_callback({
            "event": "start",
            "total_pdfs": total_pdfs,
            "processed_pdfs": 0,
            "remaining_pdfs": total_pdfs,
        })
    
    print(f"Extracting {len(pdf_path_strs)} PDFs with {max_workers} parallel workers...")
    
    def extract_single_pdf(pdf_path_str: str) -> List[Document]:
        try:
            pdf_name = os.path.basename(pdf_path_str)
            print(f"  Processing {pdf_name}...")
            docs = pdf_to_documents(
                pdf_path_str,
                include_images=include_images,
                vision_model=vision_model,
                vision_base_url=vision_base_url,
                fallback_vision_model=fallback_vision_model,
                fallback_vision_base_url=fallback_vision_base_url,
                max_images_per_pdf=max_images_per_pdf
            )
            print(f"    ✓ Extracted {len(docs)} chunks from {pdf_name}")
            return docs
        except Exception as e:
            pdf_name = os.path.basename(pdf_path_str)
            print(f"  ✗ Error extracting {pdf_name}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single_pdf, pdf_path): pdf_path for pdf_path in pdf_path_strs}
        processed_pdfs = 0
        for future in as_completed(futures):
            pdf_path = futures[future]
            docs = future.result()
            if docs:
                all_documents.extend(docs)
            processed_pdfs += 1
            if progress_callback:
                progress_callback({
                    "event": "progress",
                    "pdf_path": pdf_path,
                    "pdf_name": os.path.basename(pdf_path),
                    "processed_pdfs": processed_pdfs,
                    "total_pdfs": total_pdfs,
                    "remaining_pdfs": max(total_pdfs - processed_pdfs, 0),
                    "chunks_extracted": len(docs),
                })
    
    print(f"\n✓ Total: {len(all_documents)} chunks extracted from {len(pdf_path_strs)} PDFs.")
    if progress_callback:
        progress_callback({
            "event": "complete",
            "total_pdfs": total_pdfs,
            "processed_pdfs": total_pdfs,
            "remaining_pdfs": 0,
            "chunks_extracted": len(all_documents),
        })
    return all_documents


class MarkdownPDFDirectoryLoader:
    """
    LangChain-compatible directory loader that uses the new block-based extraction method.
    """
    def __init__(
        self,
        path: str,
        glob: str = "**/*.pdf",
        recursive: bool = True,
        include_images: bool = False, # Image extraction is simplified for now
        vision_model: str = DEFAULT_VISION_MODEL,
        vision_base_url: str | None = None,
        fallback_vision_model: str | None = None,
        fallback_vision_base_url: str | None = None,
        max_workers: int = 4,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        **kwargs  # To absorb unused arguments from the old interface
    ):
        self.path = path
        self.recursive = recursive
        self.max_workers = max_workers
        self.include_images = include_images
        self.vision_model = vision_model
        self.vision_base_url = vision_base_url
        self.fallback_vision_model = fallback_vision_model
        self.fallback_vision_base_url = fallback_vision_base_url
        self.progress_callback = progress_callback

    def load(self) -> List[Document]:
        """Load and return all documents from the directory, chunked by content block."""
        return extract_pdfs_from_directory(
            self.path,
            recursive=self.recursive,
            max_workers=self.max_workers,
            include_images=self.include_images,
            vision_model=self.vision_model,
            vision_base_url=self.vision_base_url,
            fallback_vision_model=self.fallback_vision_model,
            fallback_vision_base_url=self.fallback_vision_base_url,
            progress_callback=self.progress_callback,
        )
