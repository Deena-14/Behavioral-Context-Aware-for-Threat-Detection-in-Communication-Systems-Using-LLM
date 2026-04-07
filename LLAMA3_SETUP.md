# Llama 3 Integration — Setup Guide

This project now uses **Llama 3** as the primary LLM, matching the paper's description of a
"domain-specifically fine-tuned Llama 3 Large Language Model" for cross-verification.

---

## Architecture (Three-Tier Fallback)

```
Security Data → Primary ML Model → Llama 3 Cross-Verification → Alert
                                         ↑
                              (eliminates false positives,
                               context-aware interpretation,
                               RAG knowledge enrichment)

Tier 1: Llama 3 via Groq API      ← RECOMMENDED (free, fast, cloud)
Tier 2: Llama 3 via Ollama         ← fully local, no API key
Tier 3: Rule-based fallback        ← no setup required, always works
```

---

## Option A: Groq API (Recommended — Free)

Groq gives free access to llama3-70b-8192. No credit card needed.

1. Sign up at https://console.groq.com (free)
2. Create an API key under "API Keys"
3. Set the environment variable:

**Windows:**
```cmd
set GROQ_API_KEY=gsk_your_key_here
```

**Linux/Mac:**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

**Or add to a `.env` file in the project folder:**
```
GROQ_API_KEY=gsk_your_key_here
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the project:
```bash
python main.py
```

---

## Option B: Ollama (Fully Local — No API Key)

Ollama runs Llama 3 on your machine. Good for privacy, requires ~5GB disk.

1. Download Ollama from https://ollama.com and install it
2. Pull the Llama 3 model:
```bash
ollama pull llama3
```
3. Ollama starts automatically. Run your project normally:
```bash
python main.py
```

The code auto-detects Ollama at `http://localhost:11434`.

---

## Verification

When Llama 3 is active, you will see in the logs:
```
INFO  Llama3Client: using Groq API backend (llama3-70b-8192)
INFO  Calling Llama 3 (Llama 3 via Groq (llama3-70b-8192))…
INFO  Llama 3 cross-verification: confirmed=True, false_positive=False, adjusted_confidence=0.87
```

When using rule-based fallback (no API key):
```
INFO  No LLM backend configured. Set GROQ_API_KEY for Llama 3 (free: console.groq.com)...
INFO  Using rule-based fallback.
```

---

## What Changed in the Code

| File | Change |
|------|--------|
| `llama3_analysis.py` | **NEW** — Llama 3 client (Groq + Ollama), cross-verifier, RAG enrichment |
| `llm_analysis.py` | Updated `_call_llm()` and `perform_threat_analysis()` to use Llama 3 first |
| `config.py` | Added `LLAMA3_CONFIG` section with Groq/Ollama settings |
| `requirements.txt` | Added `groq>=0.9.0` |

---

## How Cross-Verification Works (Paper Section 3.4)

```
Primary ML model detects → threat_type, confidence, risk_level
         ↓
Llama 3 receives: primary detection + raw multimodal data
         ↓
Llama 3 outputs:
  • primary_detection_confirmed: true/false
  • false_positive: true/false  ← eliminates false alerts (93.8% reduction)
  • adjusted_confidence: 0.0-1.0
  • mitre_techniques: [T-codes]
  • verdict_reasoning: detailed explanation
         ↓
If false_positive=true → threat suppressed → alert NOT generated
If confirmed=true      → adjusted confidence used → alert generated
```

This is how the paper achieves **93.8% reduction in false positive alerts**.
