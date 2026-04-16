#!/usr/bin/env python
"""Test PDF extraction to verify pymupdf4llm is working"""

from pdf_markdown_extractor import pdf_to_documents
import os

# Test on one PDF
test_pdf = "drive_docs/_online_discovered/Federal_National/Load_20Restraint_20Guide_202004_20Full_20Copy.pdf"
print(f"Testing extraction on: {test_pdf}")

try:
    docs = pdf_to_documents(test_pdf, include_images=False)
    print(f"✓ Extracted: {len(docs)} documents")
    
    if docs:
        doc = docs[0]
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Extraction method: {doc.metadata.get('extractor', 'unknown')}")
        print(f"  First 200 chars:")
        print(f"  {doc.page_content[:200]}...")
    else:
        print("⚠ No content extracted")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
