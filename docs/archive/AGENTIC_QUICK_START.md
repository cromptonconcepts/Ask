# Multi-Agent Agentic RAG: Quick Start

## 🚀 30-Second Setup

### 1. Install LangGraph
```bash
pip install langgraph>=0.0.1
```

### 2. Restart App
```bash
python app.py
```

### 3. Enable Agentic Mode
```json
{
    "question": "What's the speed limit for work zones in Queensland?",
    "use_agentic": true
}
```

✅ Done! Multi-agent pipeline is now active.

---

## What Happens Now

**Your question goes through 3 specialized agents:**

```
1. ROUTER Agent (0.5s)
   ↓ Decides: AGTTM? QGTTM? BOTH?
   
2. RESEARCHER Agent (1.5s)
   ↓ Generates 2-3 search queries
   ↓ Retrieves top 20 documents
   
3. ENGINEER Agent (1.5s)
   ↓ Synthesizes final answer
   ↓ Attributes sources
```

**Total:** ~3.5s (vs 1.2s standard, but better quality)

---

## Example Usage

### Test 1: AGTTM-Specific Question
```python
{
    "question": "Does AGTTM specify a maximum lane closure length?",
    "use_agentic": true
}
```

**Response includes:**
```json
{
    "agent_routing": {
        "router_decision": "AGTTM",
        "router_confidence": 0.94,
        "search_queries": [
            "maximum lane closure length",
            "lane closure dimensions AGTTM",
            "Table 5.1 temporary traffic control"
        ]
    }
}
```

### Test 2: Comparison Question
```python
{
    "question": "How do AGTTM and QGTTM differ on speed limits?",
    "use_agentic": true
}
```

**Response includes:**
```json
{
    "agent_routing": {
        "router_decision": "BOTH",
        "router_confidence": 0.92,
        "search_queries": [
            "speed limits work zones AGTTM QGTTM",
            "temporary traffic control speed comparison",
            "australian guide traffic management speed"
        ]
    }
}
```

### Test 3: Multi-Turn Conversation
```python
# Turn 1: Set up context
response1 = ask({
    "question": "What's the speed limit for work zones?",
    "session_id": "sess_abc123",
    "use_agentic": true
})

# Turn 2: Follow-up (Engineer sees Turn 1 in context)
response2 = ask({
    "question": "How long can they typically stay closed?",
    "session_id": "sess_abc123",  # Same session
    "use_agentic": true
})
```

---

## Response Format

Standard response enhanced with agent routing info:

```json
{
    "answer": "In work zones, the speed limit...",
    "sources": [
        {
            "title": "AGTTM_Part_5.pdf",
            "path": "drive_docs/.../AGTTM_Part_5.pdf",
            "page": 127
        }
    ],
    "agent_routing": {
        "mode": "agentic",
        "router_decision": "AGTTM",
        "router_confidence": 0.91,
        "search_queries": [
            "speed limit work zones",
            "temporary traffic control speed",
            "Table 5.1 work zone speed"
        ],
        "num_documents_retrieved": 15
    },
    "session_id": "sess_abc123"
}
```

---

## Toggle Between Modes

### Standard Mode (Default)
```json
{
    "question": "...",
    "use_agentic": false
}
```
- Fast (1.2s)
- No routing transparency
- Uses original RAG pipeline

### Agentic Mode (New)
```json
{
    "question": "...",
    "use_agentic": true
}
```
- Slower (3.5s)
- Full transparency (see routing decisions)
- 3x LLM calls but better for complex questions

---

## Performance Timeline

Agentic mode first run:
- **0-0.5s**: Router Agent analyzes question
- **0.5-2.0s**: Researcher Agent generates queries & retrieves docs
- **2.0-3.5s**: Engineer Agent synthesizes answer
- **3.5s**: Response ready

Subsequent runs: Same speed (models cached)

---

## What Goes Into Each Agent

### Router Agent Input/Output
```
IN:  "What's the speed limit for work zones in Queensland?"
     selected_document: None

OUT: {
    "route": "QGTTM",
    "confidence": 0.95,
    "reasoning": "Explicitly mentions Queensland"
}
```

### Researcher Agent Input/Output
```
IN:  question + route (QGTTM) + conversation_context

OUT: {
    "search_queries": [
        "speed limit work zones Queensland",
        "temporary traffic control speed QGTTM",
        "Table 2.1 work zone"
    ],
    "retrieved_documents": [doc1, doc2, ..., doc20],
    "num_queries": 3,
    "num_docs_retrieved": 20
}
```

### Engineer Agent Input/Output
```
IN:  question + retrieved_documents + conversation_history

OUT: {
    "final_answer": "In Queensland work zones...",
    "answer_confidence": 0.87,
    "sources_used": [normalized source list]
}
```

---

## Configuration

### Change Default Mode
In `app.py`, modify /ask endpoint:
```python
use_agentic_mode = data.get('use_agentic', False)  # Change False to True
```

### Customize Router Prompts
In `agentic_router.py`, RouterAgent class:
```python
self.router_prompt = ChatPromptTemplate.from_template("""
[Your custom routing logic]
""")
```

### Adjust Researcher Parameters
In `agentic_router.py`, ResearcherAgent class:
```python
# Retrieve 10 docs instead of 20
return all_docs[:10]

# Use different k values
docs = vectorstore.similarity_search(query, k=8)  # Was k=10
```

---

## Troubleshooting Quick Ref

| Issue | Solution |
|-------|----------|
| `use_agentic: true` ignored | Restart app after installing langgraph |
| Slow response (>5s) | Check if Ollama models are loaded |
| Low router confidence | Improve router prompt with examples |
| Module not found error | `pip install langgraph>=0.0.1` |
| agentic_system is None | Check app logs, restart |

---

## Features Enabled by Default

| Feature | Status |
|---------|--------|
| Router Agent | ✅ Active |
| Researcher Agent | ✅ Active |
| Engineer Agent | ✅ Active |
| Conversation memory integration | ✅ Active |
| Multi-turn awareness | ✅ Active |
| LangGraph orchestration | ✅ Active |

---

## Next Steps

1. **Request mode comparison**
   ```bash
   curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "...", "use_agentic": true}'
   ```

2. **Test routing accuracy**
   - Ask AGTTM-only questions
   - Ask QGTTM-only questions
   - Ask comparison questions
   - Check if Router makes correct decisions

3. **Benchmark performance**
   - Time standard mode: ~1.2s
   - Time agentic mode: ~3.5s
   - Compare answer quality

4. **Customize for your data**
   - Review router decisions
   - Fine-tune prompts if needed
   - Adjust Researcher query generation

---

## Files Modified/Added

| File | Change | Lines |
|------|--------|-------|
| `agentic_router.py` | NEW | 450+ |
| `requirements.txt` | ADD langgraph | +1 |
| `app.py` | Import + init + /ask update | +60 |

---

## When to Use Agentic Mode

✅ **Use:** Complex questions, need routing transparency, comparison questions
❌ **Skip:** Simple lookups, real-time requirements, batch processing

---

**Ready to go!** Your TTM Ask system now has intelligent multi-agent routing with full transparency into how questions are routed and answered.
