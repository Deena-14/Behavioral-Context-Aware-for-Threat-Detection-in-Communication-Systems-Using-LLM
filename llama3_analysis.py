"""
Llama 3 Integration for Multimodal LLM-Based Cybersecurity Threat Detection
============================================================================
Implements the domain-specifically fine-tuned Llama 3 Large Language Model
described in the paper (Section 3.4) as the secondary cross-verification model.

Two backends are supported:
  1. Groq API  — cloud, free tier, fastest (recommended for most users)
                 Set env: GROQ_API_KEY=<your-key>  (https://console.groq.com)
  2. Ollama    — fully local, no API key needed
                 Install Ollama and run: ollama pull llama3

Backend priority (auto-selected):
  GROQ_API_KEY present → Groq
  Ollama running locally → Ollama
  Neither → rule-based fallback (same as llm_analysis.py)

Paper equations implemented:
  yi = F_LLM(Ti)              (Equation 8)  — Llama 3 cross-verification
  yi ∈ {0, 1}                 (Equation 9)
  Match(Ti, K) > α            (Equation 13) — RAR knowledge matching
  Ai = Alert(yi, Severity_i)  (Equation 14)
"""

import json
import logging
import os
import math
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ── Llama 3 model identifiers ─────────────────────────────────────────────────
LLAMA3_GROQ_MODEL   = "llama3-70b-8192"     # 70B via Groq (fast, free tier)
LLAMA3_GROQ_8B      = "llama3-8b-8192"      # 8B via Groq (even faster)
LLAMA3_OLLAMA_MODEL = "llama3"               # local Ollama tag

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — domain-specific fine-tuning context for Llama 3
# Mirrors the paper's description of a cybersecurity-tuned LLM
# ─────────────────────────────────────────────────────────────────────────────
LLAMA3_SYSTEM_PROMPT = """You are a cybersecurity AI assistant implementing the 
cross-verification stage of a multimodal threat detection framework. Your role 
mirrors the domain-specifically fine-tuned Llama 3 model described in the paper 
"AI Driven LLM Enhanced Multimodal Cybersecurity Threat Detection in 
Communication Networks".

Your responsibilities:
1. Cross-verify threat detections from the primary ML model to reduce false positives
2. Analyse multimodal security data: system logs, network traffic, IDS alerts, 
   DNS records, and endpoint activity
3. Provide context-aware threat interpretation using MITRE ATT&CK framework
4. Perform retrieval-augmented reasoning against cybersecurity knowledge bases
5. Generate structured JSON threat analysis reports

ALWAYS respond with valid JSON only. No markdown, no preamble, no explanation 
outside the JSON structure. Your response must match the schema provided in each 
prompt exactly.

Threat categories you recognise:
- apt_activity, malware_detected, data_exfiltration, insider_threat
- ransomware, credential_stuffing, brute_force, ddos_attack
- port_scanning, privilege_escalation, suspicious_dns, no_threat
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates for Llama 3 — cybersecurity domain-specific
# ─────────────────────────────────────────────────────────────────────────────
LLAMA3_PROMPTS = {
    "threat_analysis": """Analyse the following multimodal security data from sources: {data_sources}.

SECURITY DATA:
{activity_data}

Respond ONLY with this JSON schema:
{{
  "threat_type": "<category or no_threat>",
  "risk_level": "<critical|high|medium|low>",
  "confidence": <0.0-1.0>,
  "pattern": "<observed attack pattern description>",
  "actions": ["<remediation action 1>", "<remediation action 2>"],
  "affected_systems": ["<ip or hostname>"],
  "reasoning": "<MITRE ATT&CK based reasoning>",
  "false_positive_likelihood": <0.0-1.0>,
  "cross_verification_verdict": "<confirmed|unconfirmed|false_positive>"
}}""",

    "cross_verification": """You are cross-verifying a threat detection from the primary ML model.

PRIMARY MODEL DETECTED:
- Threat type: {threat_type}
- Confidence: {confidence}
- Risk level: {risk_level}
- Pattern: {pattern}

RAW SECURITY DATA:
{activity_data}

Cross-verify and respond ONLY with this JSON:
{{
  "primary_detection_confirmed": <true|false>,
  "cross_verified_threat_type": "<category>",
  "false_positive": <true|false>,
  "false_positive_reason": "<reason if false positive, else null>",
  "adjusted_confidence": <0.0-1.0>,
  "adjusted_risk_level": "<critical|high|medium|low>",
  "additional_indicators": ["<indicator 1>"],
  "mitre_techniques": ["<T-code>"],
  "verdict_reasoning": "<detailed reasoning>"
}}""",

    "anomaly_detection": """Detect anomalies in the following security telemetry:

DATA:
{data}

Respond ONLY with this JSON:
{{
  "threat_type": "<category or no_threat>",
  "confidence": <0.0-1.0>,
  "risk_level": "<critical|high|medium|low>",
  "pattern": "<anomaly description>",
  "actions": ["<action>"],
  "affected_systems": [],
  "reasoning": "<reasoning>"
}}""",

    "rag_enrichment": """Using your cybersecurity knowledge base, enrich this threat analysis:

THREAT TYPE: {threat_type}
INDICATORS: {indicators}
MATCHED KNOWLEDGE: {knowledge}

Respond ONLY with this JSON:
{{
  "enriched_threat_type": "<refined category>",
  "mitre_technique": "<T-code>",
  "attack_chain": ["<stage 1>", "<stage 2>"],
  "severity": "<critical|high|medium|low>",
  "recommended_remediation": ["<action>"],
  "threat_intelligence_summary": "<summary>"
}}""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Llama 3 Client — handles Groq and Ollama backends
# ─────────────────────────────────────────────────────────────────────────────
class Llama3Client:
    """
    Unified client for Llama 3 inference via Groq (cloud) or Ollama (local).
    Auto-selects backend based on available credentials.
    """

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.use_groq     = GROQ_AVAILABLE and bool(self.groq_api_key)
        self.use_ollama   = not self.use_groq and self._check_ollama()
        self._log_backend()

    def _log_backend(self):
        if self.use_groq:
            logger.info("Llama3Client: using Groq API backend (llama3-70b-8192)")
        elif self.use_ollama:
            logger.info("Llama3Client: using Ollama local backend (llama3)")
        else:
            logger.info(
                "Llama3Client: no Llama 3 backend available — "
                "set GROQ_API_KEY or start Ollama. Falling back to rule-based."
            )

    def _check_ollama(self) -> bool:
        """Ping Ollama to see if it is running locally."""
        try:
            import urllib.request
            urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self.use_groq or self.use_ollama

    def complete(self, user_prompt: str, max_tokens: int = 1024) -> Optional[str]:
        """
        Send a prompt to Llama 3 and return the raw text response.
        Returns None if no backend is available or the call fails.
        """
        if self.use_groq:
            return self._groq_complete(user_prompt, max_tokens)
        if self.use_ollama:
            return self._ollama_complete(user_prompt, max_tokens)
        return None

    def _groq_complete(self, user_prompt: str, max_tokens: int) -> Optional[str]:
        try:
            client = Groq(api_key=self.groq_api_key)
            # Try 70B first, fall back to 8B if rate-limited
            for model in [LLAMA3_GROQ_MODEL, LLAMA3_GROQ_8B]:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": LLAMA3_SYSTEM_PROMPT},
                            {"role": "user",   "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.1,   # Low temp for deterministic JSON
                    )
                    logger.info(f"Groq Llama 3 response received (model={model})")
                    return response.choices[0].message.content
                except Exception as e:
                    if "rate_limit" in str(e).lower() and model == LLAMA3_GROQ_MODEL:
                        logger.warning(f"Groq rate limit on {model}, retrying with 8B…")
                        continue
                    raise
        except Exception as e:
            logger.warning(f"Groq API error: {e}")
            return None

    def _ollama_complete(self, user_prompt: str, max_tokens: int) -> Optional[str]:
        try:
            import urllib.request, urllib.parse
            payload = json.dumps({
                "model":  LLAMA3_OLLAMA_MODEL,
                "prompt": LLAMA3_SYSTEM_PROMPT + "\n\n" + user_prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            }).encode()
            req = urllib.request.Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                logger.info("Ollama Llama 3 response received")
                return result.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama error: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Llama 3 Threat Analyzer
# ─────────────────────────────────────────────────────────────────────────────
class Llama3ThreatAnalyzer:
    """
    Implements the paper's Llama 3-based secondary verification layer.

    Role in the pipeline:
      Primary ML model → flags threat → Llama3ThreatAnalyzer cross-verifies
      → eliminates false positives → context-aware interpretation + RAR
    """

    def __init__(self):
        self.client = Llama3Client()

    def _parse_json_response(self, raw: str) -> Optional[Dict]:
        """Safely extract JSON from LLM response, stripping any markdown fences."""
        if raw is None:
            return None
        clean = raw.strip()
        # Strip ```json ... ``` wrappers
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract JSON object from mixed text
            import re
            match = re.search(r'\{[\s\S]+\}', clean)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    pass
            logger.warning("Failed to parse Llama 3 JSON response")
            return None

    def analyze_threat(self, multimodal_data: str) -> Dict[str, Any]:
        """
        Primary threat analysis — yi = F_LLM(Ti) — Equation 8.
        Llama 3 analyses multimodal security telemetry.
        """
        logger.info("Llama3: performing threat analysis (Eq. 8)…")
        prompt = LLAMA3_PROMPTS["threat_analysis"].format(
            activity_data=multimodal_data,
            data_sources="system_logs, network_traffic, ids_alerts, dns_records, endpoint_activity",
        )
        raw    = self.client.complete(prompt)
        result = self._parse_json_response(raw)

        if result is None:
            logger.warning("Llama 3 unavailable or response unparseable — returning empty result")
            return {}

        # Normalise fields
        result.setdefault("threat_type",   "no_threat")
        result.setdefault("risk_level",    "low")
        result.setdefault("confidence",    0.0)
        result.setdefault("pattern",       "")
        result.setdefault("actions",       [])
        result.setdefault("affected_systems", [])
        result.setdefault("reasoning",     "")
        result.setdefault("false_positive_likelihood", 0.0)
        result.setdefault("cross_verification_verdict", "unconfirmed")
        result["yi"] = 0 if result["threat_type"] == "no_threat" else 1
        result["llm_backend"] = "llama3_groq" if self.client.use_groq else "llama3_ollama"
        return result

    def cross_verify(
        self,
        primary_result: Dict[str, Any],
        multimodal_data: str,
    ) -> Dict[str, Any]:
        """
        Cross-verification stage: Llama 3 checks the primary ML model's output
        to eliminate false positives — key contribution from Section 3.4.

        Returns a cross_verification dict with verdict and adjustments.
        """
        if not self.client.is_available():
            logger.info("Llama 3 not available — skipping cross-verification")
            return {
                "primary_detection_confirmed": True,
                "cross_verified_threat_type":  primary_result.get("threat_type", "unknown"),
                "false_positive":              False,
                "false_positive_reason":       None,
                "adjusted_confidence":         primary_result.get("confidence", 0.0),
                "adjusted_risk_level":         primary_result.get("risk_level", "low"),
                "additional_indicators":       [],
                "mitre_techniques":            [],
                "verdict_reasoning":           "Cross-verification skipped: Llama 3 not available",
                "llm_backend":                 "unavailable",
            }

        logger.info("Llama3: cross-verifying primary model detection…")
        prompt = LLAMA3_PROMPTS["cross_verification"].format(
            threat_type    = primary_result.get("threat_type",  "unknown"),
            confidence     = primary_result.get("confidence",   0.0),
            risk_level     = primary_result.get("risk_level",   "unknown"),
            pattern        = primary_result.get("pattern",      ""),
            activity_data  = multimodal_data,
        )
        raw    = self.client.complete(prompt)
        result = self._parse_json_response(raw)

        if result is None:
            return {
                "primary_detection_confirmed": True,
                "cross_verified_threat_type":  primary_result.get("threat_type", "unknown"),
                "false_positive":              False,
                "false_positive_reason":       None,
                "adjusted_confidence":         primary_result.get("confidence", 0.0),
                "adjusted_risk_level":         primary_result.get("risk_level", "low"),
                "additional_indicators":       [],
                "mitre_techniques":            [],
                "verdict_reasoning":           "Parse error in Llama 3 response",
                "llm_backend":                 "llama3_groq" if self.client.use_groq else "llama3_ollama",
            }

        result.setdefault("primary_detection_confirmed", True)
        result.setdefault("cross_verified_threat_type", primary_result.get("threat_type"))
        result.setdefault("false_positive",              False)
        result.setdefault("false_positive_reason",       None)
        result.setdefault("adjusted_confidence",         primary_result.get("confidence", 0.0))
        result.setdefault("adjusted_risk_level",         primary_result.get("risk_level", "low"))
        result.setdefault("additional_indicators",       [])
        result.setdefault("mitre_techniques",            [])
        result.setdefault("verdict_reasoning",           "")
        result["llm_backend"] = "llama3_groq" if self.client.use_groq else "llama3_ollama"

        logger.info(
            f"Llama 3 cross-verification: confirmed={result['primary_detection_confirmed']}, "
            f"false_positive={result['false_positive']}, "
            f"adjusted_confidence={result['adjusted_confidence']}"
        )
        return result

    def detect_anomalies(self, multimodal_data: str) -> Dict[str, Any]:
        """Anomaly detection pass using Llama 3."""
        logger.info("Llama3: anomaly detection pass…")
        prompt = LLAMA3_PROMPTS["anomaly_detection"].format(data=multimodal_data)
        raw    = self.client.complete(prompt)
        result = self._parse_json_response(raw) or {}
        result.setdefault("threat_type",      "no_threat")
        result.setdefault("confidence",        0.0)
        result.setdefault("risk_level",        "low")
        result.setdefault("pattern",           "")
        result.setdefault("actions",           [])
        result.setdefault("affected_systems",  [])
        result.setdefault("reasoning",         "")
        result["llm_backend"] = "llama3_groq" if self.client.use_groq else "llama3_ollama"
        return result

    def rag_enrichment(
        self,
        threat_type: str,
        indicators: List[str],
        matched_knowledge: List[Dict],
    ) -> Dict[str, Any]:
        """
        Retrieval-Augmented Generation enrichment — Section 3.4.1, Eq. 11-13.
        Llama 3 uses retrieved knowledge-base entries to enrich threat interpretation.
        """
        logger.info(f"Llama3: RAG enrichment for '{threat_type}'…")
        prompt = LLAMA3_PROMPTS["rag_enrichment"].format(
            threat_type = threat_type,
            indicators  = json.dumps(indicators[:10]),
            knowledge   = json.dumps(matched_knowledge[:3], indent=2),
        )
        raw    = self.client.complete(prompt)
        result = self._parse_json_response(raw) or {}
        result.setdefault("enriched_threat_type",         threat_type)
        result.setdefault("mitre_technique",               "N/A")
        result.setdefault("attack_chain",                  [])
        result.setdefault("severity",                      "medium")
        result.setdefault("recommended_remediation",       [])
        result.setdefault("threat_intelligence_summary",   "")
        result["llm_backend"] = "llama3_groq" if self.client.use_groq else "llama3_ollama"
        return result

    def is_available(self) -> bool:
        return self.client.is_available()

    def backend_name(self) -> str:
        if self.client.use_groq:
            return f"Llama 3 via Groq ({LLAMA3_GROQ_MODEL})"
        if self.client.use_ollama:
            return f"Llama 3 via Ollama (local)"
        return "Llama 3 (unavailable — rule-based fallback)"
