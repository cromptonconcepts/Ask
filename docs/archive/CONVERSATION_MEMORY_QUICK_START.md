# Conversation Memory - Quick Start

## 5-Minute Setup

The conversation memory feature is **already implemented in your app.py** — no configuration needed!

---

## How to Use

### From Your HTML Frontend

**Step 1:** Store the session ID from the first response

```javascript
let sessionId = null;

async function askQuestion(question) {
  const response = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: question })
  });
  
  const data = await response.json();
  
  // Store session ID on first response
  if (!sessionId) {
    sessionId = data.session_id;
    console.log(`Conversation started: ${sessionId}`);
  }
  
  return data;
}
```

**Step 2:** Pass session ID in follow-up questions

```javascript
async function askFollowUp(question) {
  const response = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question: question,
      session_id: sessionId  // ← Include session ID
    })
  });
  
  const data = await response.json();
  return data;  // Same session_id returned
}

// Usage
await askQuestion("What are manual control speeds?");
await askFollowUp("What about in school zones?");  // LLM remembers previous answer
await askFollowUp("Can you compare that to rural areas?");  // Still has context
```

**That's it!** The LLM now automatically includes previous messages in its context.

---

## Testing It

### Test 1: Basic Multi-turn

```bash
# Terminal 1: Start your backend
python app.py

# Terminal 2: Test with curl
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are manual control speeds?"}'

# Copy the session_id from response, then:
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What about school zones?", "session_id": "abc123def456"}'

# That second response should reference the first answer!
```

### Test 2: View Conversation

```bash
# Get full conversation history
curl http://localhost:5000/conversations/abc123def456

# List all active conversations
curl http://localhost:5000/conversations
```

### Test 3: Clear Conversation

```bash
# Clear a specific conversation
curl -X DELETE http://localhost:5000/conversations/abc123def456
```

---

## What Changed in Your Code

**In app.py:**

| Component | What's New |
|-----------|-----------|
| Lines 30-34 | In-memory conversation storage |
| Lines 488-593 | 6 new memory management functions |
| Line 1415 | `build_answer()` now accepts `session_id` |
| Lines 1508-1512 | `/ask` endpoint handles session ID |
| Lines 1596-1597 | Stores Q&A in memory after each answer |
| Lines 1627-1684 | 4 new API routes for conversation management |

**No changes needed to:**
- PDF extraction
- Semantic chunking
- Embeddings
- Re-ranking
- Document retrieval

---

## Frontend Integration Examples

### React Component

```jsx
import { useState } from 'react';

export function ChatInterface() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: input,
        ...(sessionId && { session_id: sessionId })
      })
    });

    const data = await response.json();

    // Store session on first message
    if (!sessionId) {
      setSessionId(data.session_id);
    }

    // Add to message list
    setMessages([
      ...messages,
      { role: 'user', text: input },
      { role: 'assistant', text: data.answer }
    ]);

    setInput('');
  };

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i} className={msg.role}>
          {msg.text}
        </div>
      ))}
      <form onSubmit={handleSubmit}>
        <input 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a follow-up question..."
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}
```

### Vue Component

```vue
<template>
  <div class="chat">
    <div v-for="msg in messages" :key="msg.timestamp">
      <div :class="msg.role">{{ msg.text }}</div>
    </div>
    <form @submit.prevent="sendMessage">
      <input 
        v-model="input"
        placeholder="Ask a question..."
      />
      <button type="submit">Send</button>
    </form>
    <small v-if="sessionId">Session: {{ sessionId }}</small>
  </div>
</template>

<script>
export default {
  data() {
    return {
      sessionId: null,
      messages: [],
      input: ''
    };
  },
  methods: {
    async sendMessage() {
      const response = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: this.input,
          ...(this.sessionId && { session_id: this.sessionId })
        })
      });

      const data = await response.json();
      
      if (!this.sessionId) {
        this.sessionId = data.session_id;
      }

      this.messages.push(
        { role: 'user', text: this.input },
        { role: 'assistant', text: data.answer }
      );

      this.input = '';
    }
  }
};
</script>
```

---

## Common Questions

### Q: Will conversations survive server restart?
**A:** No - they're stored in memory. See CONVERSATION_MEMORY_GUIDE.md for optional SQLite persistence.

### Q: How many messages can I store?
**A:** 10 per conversation (configurable). Edit in app.py line 31: `MAX_CONVERSATION_HISTORY = 20`

### Q: Can multiple users share a session?
**A:** Yes, if they have the same session_id. For user-specific sessions, add authentication first.

### Q: Does this slow down queries?
**A:** Minimal (+20-50ms for context formatting). See performance table in CONVERSATION_MEMORY_GUIDE.md

### Q: How do I delete a conversation?
**A:** `DELETE /conversations/{session_id}` or call `clear_conversation(session_id)`

### Q: What if I don't pass a session_id?
**A:** A new session is auto-generated and returned in the response. You can ignore it for stateless queries.

---

## API Summary

### 5 Endpoints

```
POST   /conversations/start              → Create new session
GET    /conversations                    → List all sessions
GET    /conversations/<session_id>       → Get conversation history
DELETE /conversations/<session_id>       → Clear conversation
POST   /ask (with session_id param)      → Ask question with context
```

---

## Next Steps

1. **Update your HTML** - Add session ID handling (see examples above)
2. **Test multi-turn** - Ask follow-up questions and verify context
3. **Monitor memory** - Check `/conversations` endpoint to see active sessions
4. **Tune if needed** - Adjust `MAX_CONVERSATION_HISTORY` if needed

---

## Documentation

- **CONVERSATION_MEMORY_GUIDE.md** - Complete API reference & examples
- **CONVERSATION_MEMORY_IMPLEMENTATION.md** - Technical implementation details
- **QUICK_REFERENCE.md** - General quick reference

---

## Example Session Flow

```
Client                          Server
  │
  ├─ POST /ask                    │
  │  {"question": "speeds?"}      │
  │                               ├─ Generate session_id: "a1b2c3d4"
  ├──────────────────────────────→├─ Answer question
  │                               ├─ Store: HUMAN: "speeds?"
  │                               ├─ Store: AI: "speeds are..."
  │  ← {"answer": "...",           │
  │     session_id: "a1b2c3d4"}    │
  │←─────────────────────────────┤
  │
  ├─ POST /ask                    │
  │  {"question": "school?",      │
  │   "session_id": "a1b2c3d4"}   │
  │                               ├─ Load conversation history
  │                               ├─ Format: "HUMAN: speeds?"
  │                               ├─          "AI: speeds are..."
  ├──────────────────────────────→├─ Pass to LLM with context
  │                               ├─ Answer with context awareness
  │                               ├─ Store: HUMAN: "school?"
  │                               ├─ Store: AI: "in schools..."
  │  ← {"answer": "in schools...", │
  │     session_id: "a1b2c3d4"}    │
  │←─────────────────────────────┤
  │
  ├─ DELETE /conversations...    │
  │  "/a1b2c3d4"                  │
  │                               ├─ Clear all messages
  ├──────────────────────────────→├─ Session history erased
  │  ← {"messages_deleted": 4}    │
  │←─────────────────────────────┤
```

---

## Troubleshooting

**Problem:** Session not found error

```javascript
if (response.status === 404) {
  // Session expired, start new one
  sessionId = null;
  await askQuestion(input);
}
```

**Problem:** LLM not referencing previous messages

- Check `session_id` is being passed to `/ask`
- Verify `GET /conversations/{id}` shows messages
- Check server logs for errors

**Problem:** Memory usage growing

- Reduce `MAX_CONVERSATION_HISTORY` 
- Reduce `MAX_CONVERSATIONS`
- Call `DELETE /conversations/{id}` for old conversations

---

## You're Ready! 🎉

Your TTM Ask system now supports full multi-turn conversations:
- ✅ Automatic context retention
- ✅ Session management
- ✅ Conversation history API
- ✅ Zero configuration needed

Start using it immediately with your existing frontend!
