# Multimodal Image Extraction - Quick Start

## 30-Second Setup

```bash
# 1. Pull a vision model (one-time)
ollama pull qwen2-vl:7b

# 2. Start Ollama server (keep running)
ollama serve

# 3. In another terminal, your app auto-uses it:
python app.py
```

**That's it!** Image extraction enables automatically on first index rebuild.

---

## What Happens Automatically

When you start the app with PDFs containing images:

```
Extracting drive_docs/...
  Extracting document.pdf...
  Extracting images and generating descriptions...
    Added 3 image descriptions to index
  ✓ Extracted markdown from document.pdf
```

Image descriptions:
```
[Image from page 5 - 1024x768px]
Description: Diagram showing left lane closure with 15-meter taper...
Image size: 1024x768 pixels
```

These are automatically:
- Embedded with text content
- Made searchable
- Included in conversation context

---

## Test It

Query your app with image-related terms:

```javascript
// Browser/App Query:
"Show me a diagram of lane closure"
"What does the traffic control diagram look like?"
"Explain the manual control setup"

// Results:
Shows image descriptions + related text guidance
```

---

## Configuration (Optional)

In `app.py`, around line 370 in `initialize_or_reload_index()`:

```python
# Enable/disable or change model
loaded_documents, _ = extract_pdfs_from_directory(
    DOCS_DIR,
    include_images=True,            # False to disable
    vision_model="qwen2-vl:7b",     # Change model
    max_images_per_pdf=5             # More = slower indexing
)
```

### Try Different Models

```python
# Fast (13B)
vision_model="bakllava:13b"

# Detailed (34B)
vision_model="llava:34b"

# Recommended (7B - trades speed for quality)
vision_model="qwen2-vl:7b"
```

---

## Models Available

| Model | Size | Speed | Quality | Install |
|-------|------|-------|---------|---------|
| bakllava:13b | 7.4GB | ⭐⭐⭐⭐ Fast | ⭐⭐⭐ Good | `ollama pull bakllava` |
| qwen2-vl:7b | 4.7GB | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Great | `ollama pull qwen2-vl` |
| llava:34b | 20GB | ⭐⭐ Slow | ⭐⭐⭐⭐ Excellent | `ollama pull llava` |

**Recommended:** qwen2-vl:7b (best for technical diagrams)

---

## Performance

### First Run (One-time)
```
5 PDFs, 2 images each:
- Markdown extraction: 2 min
- Image extraction + vision: 10-15 min
- Semantic chunking: 2 min
- Embedding: 5 min
- Total: ~20-25 minutes

50 PDFs, 3 images each:
- Total: ~3-4 hours (first time only)
```

### Subsequent Runs
```
Cached embeddings: 10-20 seconds (fast)
```

### Query Time
```
Same as before: ~1200ms
(Image descriptions already embedded, no extra compute)
```

---

## Troubleshooting

### "Vision model not found"
```bash
ollama pull qwen2-vl:7b
ollama serve
```

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Should show "Listening on 127.0.0.1:11434"
```

### Images Still Not Extracted
1. Verify PDF has images: Open in Adobe Reader, check for diagrams
2. Check logs during indexing for "Extracting images..."
3. Try test command: `python multimodal_image_extractor.py your_pdf.pdf`
4. Disable and re-enable: Set `include_images=False` then back to `True`

### Slow First Indexing
**Expected!** Vision inference takes time. Grab coffee ☕

---

## Features Enabled

✅ **Image Extraction**
- Automatically pulls all diagrams from PDFs
- Filters out tiny artifacts (< 50×50px)

✅ **Vision Descriptions**  
- Generates captions using Ollama multimodal models
- Example: "Traffic control device setup showing..."

✅ **Semantic Integration**
- Descriptions merged with document markdown
- Treated like regular text during chunking/embedding

✅ **Full-Text Search**
- Find diagrams by their content
- Search: "lane closure diagram" → finds images

✅ **Conversation Context**
- Image descriptions included in multi-turn conversations
- LLM can reference: "As shown in the diagram from earlier..."

---

## Documentation

- **Full Guide:** [MULTIMODAL_IMAGE_EXTRACTION_GUIDE.md](MULTIMODAL_IMAGE_EXTRACTION_GUIDE.md)
- **Technical:** See `multimodal_image_extractor.py` source code
- **Tests:** `python multimodal_image_extractor.py path/to/pdf.pdf`

---

## Example

PDF with 3 images → Auto-processed:

```
Original PDF:
- Page 5: Lane closure diagram
- Page 12: Traffic control setup
- Page 18: Manual device diagram

After indexing:
- All 3 images extracted
- Descriptions generated ("Diagram of a left lane closure with 15m taper...")
- Embedded in knowledge base
- Searchable: "What does the lane closure diagram show?"
```

---

## System Ready!

Your TTM Ask system now understands diagrams and images through:
- ✅ Multimodal vision models (qwen2-vl, llava, bakllava)
- ✅ Automatic image extraction during indexing
- ✅ Semantic embedding of visual content
- ✅ Full-text search on images
- ✅ Context injection in conversations

**No additional setup needed!** Just ensure Ollama vision model is pulled and you're good to go.

```bash
# One-time:
ollama pull qwen2-vl:7b

# Keep running:
ollama serve

# Your app handles the rest!
```
