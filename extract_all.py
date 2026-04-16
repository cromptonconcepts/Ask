#!/usr/bin/env python
"""Extract all PDFs and show summary"""

from pdf_markdown_extractor import extract_pdfs_from_directory

print("Extracting all PDFs from drive_docs...")
print("-" * 60)

docs = extract_pdfs_from_directory('drive_docs', recursive=True, include_images=False)

print("\n" + "=" * 60)
print(f"✓ EXTRACTION COMPLETE")
print(f"  Total documents: {len(docs)}")
print("=" * 60)

if docs:
    print("\nDocuments extracted:")
    from pathlib import Path
    import os
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'unknown')
        filename = os.path.basename(source)
        extractor = doc.metadata.get('extractor', 'unknown')
        size = len(doc.page_content)
        print(f"  [{i:2d}] {filename}")
        print(f"        Extraction: {extractor}")
        print(f"        Size: {size:,} chars")
