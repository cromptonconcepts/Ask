# TTM Ask

TTM Ask is a traffic management knowledge assistant for Australian standards and guidance documents. It supports markdown-preserving PDF extraction, semantic chunking, stronger retrieval embeddings, reranking, conversation memory, multimodal diagram understanding, and optional multi-agent routing.

## Deployment Modes

TTM Ask now supports two ways of running.

### 1. Local desktop mode

Use this when each user runs their own copy on their own Windows PC.

- backend runs on that PC
- Ollama runs on that PC
- Chroma index is stored on that PC
- easiest setup path

### 2. Shared network/web mode

Use this when one server hosts the app for many users.

- backend runs on a server
- Ollama runs on that server
- users open the app from a browser using the server URL
- one shared document corpus and index

## Local Install For Other Users

### Easiest path

1. Copy the whole project folder to the user machine.
2. Double-click Start TTM Ask.vbs.
3. On first run, the setup script will:
   - create a Python virtual environment
   - install Python packages
   - install Ollama if possible
   - pull the required models
4. On a clean machine, a visible PowerShell setup window appears so users can see progress.
5. Wait for setup to complete.
6. The app opens in the browser.

The launcher now opens the local app URL when the backend is ready, so users land on the live local service instead of a disconnected HTML file. Later launches stay quiet once setup has completed.

### First-run downloads

The first setup downloads several large items:

- Python dependencies
- qwen2.5:7b
- nomic-embed-text
- qwen2.5vl

This can take time on the first machine run.

If a model is already present on the machine, setup skips that download.

## Manual Local Setup

1. Install Python 3.10 or newer.
2. Install Ollama from https://ollama.com/download
3. Open PowerShell in the project folder.
4. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

5. Start the app:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_ttm_ask.ps1
```

## Windows Installer Flow

This repository now includes a real installer build flow using Inno Setup.

### Build the installer

1. Install Inno Setup 6.
2. Open PowerShell in the project folder.
3. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build\build_installer.ps1
```

4. The installer EXE is generated in the `dist` folder.

### Build full distribution package

Use this when you want both an installer and a portable zip for sending to other users.

1. Open PowerShell in the project folder.
2. Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\build\build_release.ps1
```

3. Artifacts are produced in the `dist` folder:
   - `Crompton_AI-Installer.exe`

Portable package is produced in the `portable` folder:
   - `Crompton_AI-Portable.zip`

The installer is recommended for normal users. The portable zip is useful for environments where installer execution is restricted.

## Folder Separation

### 1. Local run (main app folder)

Keep and run from the project root:

- `Start TTM Ask.vbs`
- `Start Crompton_AI.vbs` (legacy shortcut kept for compatibility)
- `run_ttm_ask.ps1`
- `setup_windows.ps1`
- `launch.bat`
- `app.py`, `launcher.py`, `ttm_ask.html`, `requirements.txt`
- `drive_docs/`, `chroma_db/`, `.venv/`, `assets/`, `logs/`

### 2. Portable / distribution

- `dist/Crompton_AI-Installer.exe`
- `portable/Crompton_AI-Portable.zip`

Internal design and migration notes are archived under `docs/archive/` to keep the root folder clean.
Build-only files are stored under `build/`.

### What the installer does

- copies the app into LocalAppData
- creates Start Menu and Desktop shortcuts named `TTM Ask`
- runs setup_windows.ps1 after install
- installs Python requirements and Ollama models
- starts the app at the end of installation
- creates data folders (`drive_docs`, `chroma_db`, `logs`) in the installed app directory

## Shared Network/Web Deployment

The frontend no longer assumes `localhost` when it is served over HTTP. When loaded from a server, it uses the same origin for API calls.

### Recommended architecture

Server:

- Windows Server or Linux VM
- Python 3.10+
- Ollama installed on the same server
- shared `drive_docs` folder on the server
- TTM Ask served with Waitress

Users:

- browse to the server URL
- example: `http://server-name:5000/`

### Environment variables

Set these before starting the app in shared mode:

```powershell
$env:TTM_ASK_HOST = '0.0.0.0'
$env:TTM_ASK_PORT = '5000'
$env:TTM_ASK_DEBUG = '0'
$env:TTM_ASK_CORS_ORIGINS = 'http://server-name:5000,https://your-domain.example'
$env:LOCAL_OLLAMA_BASE_URL = 'http://localhost:11434'
$env:CLOUD_OLLAMA_BASE_URL = 'https://your-ollama-endpoint.example'
$env:LOCAL_OLLAMA_CHAT_MODEL = 'qwen2.5:7b'
$env:CLOUD_OLLAMA_CHAT_MODEL = 'qwen2.5:7b'
$env:LOCAL_OLLAMA_EMBEDDING_MODEL = 'nomic-embed-text'
$env:CLOUD_OLLAMA_EMBEDDING_MODEL = 'nomic-embed-text'
$env:LOCAL_OLLAMA_VISION_MODEL = 'qwen2.5vl'
$env:CLOUD_OLLAMA_VISION_MODEL = 'qwen2.5vl'
$env:OLLAMA_KEEP_ALIVE = '-1'
```

If `CLOUD_OLLAMA_BASE_URL` is set, the app tries cloud Ollama first for chat, embeddings, and review-image extraction, then falls back to local Ollama when the cloud endpoint is unavailable or does not host the required model.

Cloud Ollama supports image understanding only when the configured cloud model is a vision-capable model such as `qwen2.5vl`.

### Start shared mode with Waitress

1. Run setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

2. Set server binding:

```powershell
$env:TTM_ASK_HOST = '0.0.0.0'
```

3. Start the backend:

```powershell
.\.venv\Scripts\python.exe .\serve_ttm_ask.py
```

4. Open firewall access to the selected port.
5. Users browse to `http://server-name:5000/`

### Recommended production extras

Put the app behind IIS, Nginx, or another reverse proxy so you can add:

- HTTPS
- authentication
- logging
- friendly DNS names
- rate limiting

## Local Versus Shared Behavior

### Local mode

- `ttm_ask.html` can be opened directly
- Start Services button is available
- launcher can start backend services locally

### Shared mode

- users open `http://server:5000/`
- frontend talks to the same host that served it
- Start Services button stays hidden for users

## Updating Documents

1. Add PDFs under `drive_docs`.
2. Restart the backend or call the reload endpoint.
3. The app rebuilds the index when source files change.

## Structured TGS/TMP Review

The backend now supports a structured review workflow for:

- TGS vs TMP alignment
- compliance checks against indexed Austroads, federal/national, and Queensland guidance
- discrepancy detection inside or across TGS, TMP, and CTMP files
- additional reviewer checks such as pedestrian access, emergency access, sequencing, missing dimensions, and conflicting notes

### API route

POST `/review-documents`

Example JSON body:

```json
{
   "tgs_path": "C:/path/to/TGS.pdf",
   "tmp_path": "C:/path/to/TMP.pdf",
   "ctmp_path": "C:/path/to/CTMP.pdf",
   "additional_focus": [
      "bus stop impacts",
      "night works"
   ]
}
```

The response includes structured findings, a consolidated recommendation list, and a markdown-formatted review report.

### CLI usage

```powershell
.\.venv\Scripts\python.exe .\traffic_plan_reviewer.py --tgs "C:\path\to\TGS.pdf" --tmp "C:\path\to\TMP.pdf" --markdown-output .\review_report.md --json-output .\review_report.json
```

## Files Added For Distribution And Deployment

- `setup_windows.ps1` - first-time setup for new user machines
- `run_ttm_ask.ps1` - portable local launcher
- `serve_ttm_ask.py` - production server entrypoint using Waitress
- `build/TTM_Ask_Setup.iss` - Inno Setup installer definition
- `build/build_installer.ps1` - builds the installer EXE
- `build/build_release.ps1` - builds installer and portable zip

## Troubleshooting

### Ollama is missing

Install it manually from:

https://ollama.com/download

Then run:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama pull qwen2.5vl
```

### Scripts are blocked by execution policy

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

### Shared users cannot reach the app

Check:

- `TTM_ASK_HOST` is set to `0.0.0.0`
- firewall allows the selected port
- server name or IP is reachable from user machines
- Ollama is running on the server

## Recommended Next Steps

For a stronger shared rollout, the next logical upgrades are:

1. persistent conversation storage instead of in-memory only
2. authentication for shared users
3. Windows Service registration for backend startup
4. admin UI for reload/index health