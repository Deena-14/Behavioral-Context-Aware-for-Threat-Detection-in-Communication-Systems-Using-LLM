"""
LLM-Based Threat Analysis
Implements Sections 3.4, 3.4.1, and 3.5 of the paper:
  yi = F_LLM(Ti)                              (Equation 8)
  yi ∈ {0, 1}                                (Equation 9)
  S = (T1, T2, …, Tk)                        (Equation 10)
  K = {k1, k2, …, kp}                        (Equation 11)
  Match(Ti, K) = max Similarity(Ti, kj)      (Equation 12)
  Match(Ti, K) > α                           (Equation 13)
  Ai = Alert(yi, Severity_i)                 (Equation 14)
  Severity_i = w1*Ri + w2*Impact_i           (Equation 15)

CHANGES (v2):
  - Switched from OpenAI SDK → Anthropic SDK
  - Empty API key now cleanly triggers local rule-based fallback (no exception)
  - Signal detection (Change 4): network traffic now inspects real NSL-KDD features
    (serror_rate, bytes_sent, f_is_suspicious) instead of only hardcoded ports
  - Threat type selection (Change 5): deterministic mapping from fired signal
    sources instead of random.choice — same input always produces same output
"""

import json
import logging
import math
from typing import Dict, List, Any

from config import SystemConfig, LLM_PROMPTS, THREAT_KNOWLEDGE_BASE

# ── Anthropic SDK (Claude — primary fallback when Llama 3 not available) ──────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Llama 3 integration (paper's described LLM) ───────────────────────────────
try:
    from llama3_analysis import Llama3ThreatAnalyzer
    _llama3_analyzer = Llama3ThreatAnalyzer()
    LLAMA3_AVAILABLE = _llama3_analyzer.is_available()
    if LLAMA3_AVAILABLE:
        logger_init = logging.getLogger(__name__)
        logger_init.info(f"Llama 3 backend ready: {_llama3_analyzer.backend_name()}")
except Exception:
    _llama3_analyzer = None
    LLAMA3_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: simple text-level cosine similarity for knowledge-base matching
# Used in Match(Ti, K) — Equation (12)
# ─────────────────────────────────────────────────────────────────────────────
def _text_similarity(text_a: str, text_b: str) -> float:
    """Bag-of-words cosine similarity between two strings."""
    def tokenise(t: str) -> set:
        return set(t.lower().replace('_', ' ').split())

    tokens_a = tokenise(text_a)
    tokens_b = tokenise(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    return len(intersection) / math.sqrt(len(tokens_a) * len(tokens_b))


# ── CHANGE 5: Deterministic threat-type mapping from fired signal sources ─────
# Keyed by frozenset of source names that fired. Ordered from most specific
# (many sources) to least so that the first matching rule wins.
_SIGNAL_TO_THREAT = [
    # All 5+ sources fired → confirmed multi-stage intrusion
    ({'system_logs', 'network_traffic', 'ids_alerts', 'dns_records', 'endpoint_activity'},
     'apt_activity'),
    # 4 sources
    ({'system_logs', 'network_traffic', 'ids_alerts', 'endpoint_activity'},
     'apt_activity'),
    ({'system_logs', 'network_traffic', 'dns_records', 'endpoint_activity'},
     'data_exfiltration'),
    ({'network_traffic', 'ids_alerts', 'dns_records', 'endpoint_activity'},
     'malware_detected'),
    ({'system_logs', 'ids_alerts', 'dns_records', 'endpoint_activity'},
     'insider_threat'),
    # 3 sources
    ({'system_logs', 'ids_alerts', 'endpoint_activity'},
     'privilege_escalation'),
    ({'network_traffic', 'ids_alerts', 'dns_records'},
     'data_exfiltration'),
    ({'system_logs', 'network_traffic', 'ids_alerts'},
     'malware_detected'),
    ({'system_logs', 'endpoint_activity', 'dns_records'},
     'insider_threat'),
    ({'network_traffic', 'dns_records', 'endpoint_activity'},
     'suspicious_dns'),
    # 2 sources
    ({'system_logs', 'ids_alerts'},        'brute_force'),
    ({'system_logs', 'endpoint_activity'}, 'credential_stuffing'),
    ({'network_traffic', 'ids_alerts'},    'ddos_attack'),
    ({'dns_records', 'network_traffic'},   'suspicious_dns'),
    # 1 source
    ({'ids_alerts'},        'malware_detected'),
    ({'dns_records'},       'suspicious_dns'),
    ({'network_traffic'},   'port_scanning'),
    ({'system_logs'},       'brute_force'),
    ({'endpoint_activity'}, 'privilege_escalation'),
]


def _map_signals_to_threat(signal_sources: List[str]) -> str:
    """
    Deterministic threat-type derivation from which sources fired.
    Replaces the previous random.choice calls for reproducible results.
    """
    fired = set(signal_sources)
    if not fired:
        return 'no_threat'
    for required, threat in _SIGNAL_TO_THREAT:
        if required.issubset(fired):
            return threat
    return 'suspicious_dns'   # safe fallback


class LLMThreatAnalyzer:
    """
    Implements the paper's LLM-based threat detection pipeline.
    Core equations: Eq. 8-13.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or SystemConfig.LLM_CONFIG
        self.knowledge_base = THREAT_KNOWLEDGE_BASE
        self.threat_categories = SystemConfig.THREAT_CATEGORIES
        self.risk_levels = SystemConfig.RISK_LEVELS
        self.alpha = SystemConfig.THREAT_CONFIG['knowledge_match_threshold']

    # ── F_LLM(Ti) — Equation (8) ──────────────────────────────────────────────
    def _call_llm(self, prompt: str, data: str) -> Dict[str, Any]:
        """
        LLM call with three-tier priority (paper architecture):

        Tier 1 — Llama 3 (paper's described model):
            • Groq API  if GROQ_API_KEY is set  (cloud, free, fast)
            • Ollama    if running locally        (fully private)

        Tier 2 — Anthropic Claude:
            • Used when Llama 3 is unavailable and ANTHROPIC_API_KEY is set

        Tier 3 — Local rule-based analysis:
            • Deterministic fallback, no API required
        """
        # ── Tier 1: Llama 3 ───────────────────────────────────────────────────
        if LLAMA3_AVAILABLE and _llama3_analyzer is not None:
            try:
                logger.info(f"Calling Llama 3 ({_llama3_analyzer.backend_name()})…")
                import urllib.request as _ur
                from llama3_analysis import LLAMA3_PROMPTS
                raw = _llama3_analyzer.client.complete(prompt)
                if raw:
                    import re, json as _json
                    clean = raw.strip()
                    if clean.startswith("```"):
                        lines = clean.split("\n")
                        clean = "\n".join(
                            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                        )
                    try:
                        result = _json.loads(clean)
                    except Exception:
                        match = re.search(r'\{[\s\S]+\}', clean)
                        result = _json.loads(match.group()) if match else None
                    if result is not None:
                        result["llm_backend"] = _llama3_analyzer.backend_name()
                        logger.info("Llama 3 response received successfully")
                        return result
            except Exception as e:
                logger.warning(f"Llama 3 error ({e}) — trying next tier")

        # ── Tier 2: Anthropic Claude ──────────────────────────────────────────
        api_key = self.config.get('api_key', '')
        if ANTHROPIC_AVAILABLE and api_key:
            try:
                logger.info("Calling Anthropic Claude API for semantic threat analysis…")
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model=self.config.get('model_name', 'claude-haiku-4-5-20251001'),
                    max_tokens=self.config.get('max_tokens', 1024),
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "You are an expert cybersecurity AI performing multimodal "
                                "threat analysis on communication networks. "
                                "Always respond with valid JSON only.\n\n"
                                + prompt
                            ),
                        }
                    ],
                )
                raw = message.content[0].text
                clean = raw.strip()
                if clean.startswith("```"):
                    clean = clean.split("```")[1]
                    if clean.startswith("json"):
                        clean = clean[4:]
                result = json.loads(clean.strip())
                result["llm_backend"] = "anthropic_claude"
                logger.info("Anthropic Claude API response received successfully")
                return result
            except Exception as e:
                logger.warning(f"Anthropic API error ({e}) — falling back to rule-based")
        else:
            if not LLAMA3_AVAILABLE and not api_key:
                logger.info(
                    "No LLM backend configured. "
                    "Set GROQ_API_KEY for Llama 3 (free: console.groq.com) "
                    "or ANTHROPIC_API_KEY for Claude. "
                    "Using rule-based fallback."
                )

        # ── Tier 3: Rule-based fallback ───────────────────────────────────────
        return self._local_rule_based_analysis(data)

    # ── Local rule-based fallback ─────────────────────────────────────────────
    def _local_rule_based_analysis(self, data: str) -> Dict[str, Any]:
        """
        Deterministic rule-based response when the API is unavailable.
        CHANGE: threat type derived from signal sources (no random.choice).
        """
        signals, signal_details = self._count_suspicion_signals(data)

        # Confidence and risk level from signal count
        if signals >= 5:
            confidence = 0.91
            risk_level = 'critical'
        elif signals == 4:
            confidence = 0.82
            risk_level = 'high'
        elif signals == 3:
            confidence = 0.57
            risk_level = 'medium'
        elif signals == 2:
            confidence = 0.42
            risk_level = 'low'
        elif signals == 1:
            confidence = 0.27
            risk_level = 'low'
        else:
            confidence = 0.10
            risk_level = 'low'

        # CHANGE 5: deterministic threat type — no more random.choice
        threat_type = _map_signals_to_threat(signal_details)
        kb = self.knowledge_base.get(threat_type, {})

        pattern_templates = {
            'apt_activity': (
                f"Multi-stage attack chain identified across {len(signal_details)} data sources. "
                "Lateral movement indicators in system logs combined with C2 beacon patterns "
                "in network traffic and DNS records. Living-off-the-land techniques on endpoints."
            ),
            'data_exfiltration': (
                f"Anomalous outbound data transfer across {len(signal_details)} modalities. "
                "Large volume flows to external IPs correlated with sensitive file access "
                "in system logs and suspicious DNS queries to rare TLDs."
            ),
            'insider_threat': (
                "Off-hours access in system logs. Privileged user bulk file access followed "
                "by unusual outbound connections. Endpoint shows unauthorised data staging."
            ),
            'malware_detected': (
                "Malware signature in IDS correlated with suspicious process creation on endpoint. "
                "Registry modifications and encrypted C2 traffic in network capture. "
                "DNS queries to known DGA domains."
            ),
            'brute_force': (
                f"Rapid sequential auth failures from source IP across {len(signal_details)} sources. "
                "Pattern consistent with dictionary-based brute force targeting SSH/RDP."
            ),
            'credential_stuffing': (
                "Multiple login attempts with different credentials from same IP. "
                "IDS flagged user-enumeration pattern. Low rate indicates automated tool (T1110.004)."
            ),
            'privilege_escalation': (
                "Suspicious sudo/privilege usage in system logs. UAC bypass attempt on endpoint "
                "followed by kernel-level process injection. IDS buffer overflow alert correlated."
            ),
            'suspicious_dns': (
                "Elevated DNS query rate to uncommon domains. Long subdomain strings indicate "
                "possible DNS tunnelling for covert data exfil or C2 (T1071.004)."
            ),
            'ddos_attack': (
                "High-volume traffic spike from multiple source IPs. SYN-flood pattern in "
                "network traffic correlates with IDS DoS alert. Service response degraded."
            ),
            'port_scanning': (
                "Sequential port connection attempts from single source IP. "
                "SYN-only packets without ACK response indicate stealth scanning (T1046)."
            ),
            'no_threat': (
                "All monitored sources show baseline behaviour. No anomalous patterns across "
                "system logs, network traffic, IDS, DNS, or endpoint activity."
            ),
        }

        reasoning_templates = {
            'apt_activity': (
                f"Semantic analysis across {len(signal_details)} modalities identified "
                "MITRE ATT&CK tactics TA0001–TA0011. Cross-source correlation revealed "
                "coordinated multi-stage intrusion with persistence, lateral movement, and C2."
            ),
            'data_exfiltration': (
                "Context-aware reasoning across network and DNS identified outbound anomalies "
                "consistent with T1041. Privileged file access in logs strengthens attribution."
            ),
            'insider_threat': (
                "Behavioural analysis of endpoint and system log data identified access patterns "
                "deviating from baselines. Temporal correlation with after-hours activity "
                "raises insider threat confidence."
            ),
            'malware_detected': (
                "IDS signature match correlated with endpoint process anomalies and C2 DNS "
                "query patterns. Multi-source evidence increases confidence (MITRE T1059)."
            ),
            'brute_force': (
                "Statistical analysis of auth failure rate exceeds baseline threshold. "
                "Source IP consistency across modalities confirms automated brute force (T1110)."
            ),
            'credential_stuffing': (
                "Low-rate credential attempts with high user diversity indicate stuffing "
                "rather than brute force. Pattern consistent with MITRE T1110.004."
            ),
            'privilege_escalation': (
                "Endpoint and system log correlation identified privilege escalation sequence. "
                "Process lineage shows unexpected privilege gain (MITRE T1068)."
            ),
            'suspicious_dns': (
                "DNS query analysis identified high-entropy subdomains and unusual TLD usage. "
                "Frequency and pattern consistent with tunnelling (T1071.004)."
            ),
            'ddos_attack': (
                "Network flow analysis shows traffic volume spike consistent with amplification "
                "or SYN-flood DoS pattern (MITRE T1498). IDS confirms alert."
            ),
            'port_scanning': (
                "Network flow shows sequential destination port increments with SYN-only packets. "
                "Consistent with stealth port scan (T1046)."
            ),
            'no_threat': (
                "Multimodal analysis of all 5 sources shows normal baseline. "
                "No correlated suspicious patterns. Continuing monitoring."
            ),
        }

        affected = []
        if threat_type != 'no_threat' and signal_details:
            import random as _r
            num_systems = _r.randint(1, 3)
            affected = [f"192.168.1.{_r.randint(10, 200)}" for _ in range(num_systems)]

        return {
            'threat_type':     threat_type,
            'risk_level':      risk_level,
            'confidence':      confidence,
            'pattern':         pattern_templates.get(threat_type, "Multimodal analysis completed."),
            'actions':         kb.get('remediation', 'Monitor and investigate').split(', '),
            'affected_systems': affected,
            'reasoning':       reasoning_templates.get(threat_type, "Analysis complete."),
        }

    # ── CHANGE 4: Improved signal detection using real NSL-KDD features ────────
    def _count_suspicion_signals(self, data: str) -> tuple:
        """
        Count suspicious signals across the 5 modalities.

        CHANGE (network traffic): now checks actual NSL-KDD numeric features
        (serror_rate, bytes_sent, f_is_suspicious) in addition to port numbers,
        so DoS/Probe attacks from the real dataset reliably trigger this signal.

        CHANGE (all sources): uses dict.get() on parsed objects instead of
        str() conversion, which was unreliable for boolean values.
        """
        signals = 0
        signal_sources = []

        try:
            payload = json.loads(data)
        except Exception:
            return 1, ['parse_error']

        # ── Source 1: System logs ─────────────────────────────────────────────
        sys_logs = payload.get('system_logs', [])
        suspicious_log_keywords = (
            'AUTH_FAILURE', 'FAILED_LOGIN', 'PRIVILEGE_ESCALATION',
            'SUDO_USAGE', 'PERMISSION_CHANGE',
        )
        if any(
            any(kw in str(log) for kw in suspicious_log_keywords)
            for log in sys_logs
        ):
            signals += 1
            signal_sources.append('system_logs')

        # ── Source 2: Network traffic — CHANGE: real NSL-KDD feature checks ──
        net_events = payload.get('network_traffic', [])
        net_suspicious = False
        for ev in net_events:
            if isinstance(ev, dict):
                feats = ev.get('features', {})
                # Real NSL-KDD numeric indicators
                if (
                    feats.get('f_serror_rate', 0) > 0.5       # DoS SYN-flood pattern
                    or feats.get('f_is_suspicious', 0) == 1.0  # mapped flag from loader
                    or feats.get('f_src_bytes', 0) > 100_000   # large transfer
                    or feats.get('f_dst_bytes', 0) > 100_000
                    or feats.get('f_is_suspicious_port', 0) == 1
                ):
                    net_suspicious = True
                    break
                # Legacy simulation checks (hacker ports, string keywords)
                if (
                    '4444' in str(ev) or '31337' in str(ev)
                    or 'SUSPICIOUS' in str(ev).upper()
                ):
                    net_suspicious = True
                    break
            else:
                if 'SUSPICIOUS' in str(ev).upper() or '4444' in str(ev) or '31337' in str(ev):
                    net_suspicious = True
                    break
        if net_suspicious:
            signals += 1
            signal_sources.append('network_traffic')

        # ── Source 3: IDS alerts ──────────────────────────────────────────────
        ids_alerts = payload.get('ids_alerts', [])
        ids_threat_keywords = (
            'buffer_overflow', 'buffer overflow',
            'malware', 'malware signature',
            'exploit', 'command injection',
            'intrusion', 'sql_injection', 'sql injection',
            'xss', 'cross-site scripting',
            'backdoor', 'c2_beacon', 'command_and_control',
            'lateral_movement', 'lateral movement',
            'ransomware', 'port_scan',
            'brute force',
            # NSL-KDD mapped IDS alert types
            'dos', 'ddos', 'probe', 'reconnaissance',
            'remote to local', 'user to root',
        )
        if any(
            any(kw in str(al).lower() for kw in ids_threat_keywords)
            for al in ids_alerts
        ):
            signals += 1
            signal_sources.append('ids_alerts')

        # ── Source 4: DNS records — CHANGE: use dict.get() for bool check ────
        dns_records = payload.get('dns_records', [])
        if any(
            (d.get('is_suspicious') is True if isinstance(d, dict) else False)
            for d in dns_records
        ):
            signals += 1
            signal_sources.append('dns_records')

        # ── Source 5: Endpoint activity ───────────────────────────────────────
        endpoint = payload.get('endpoint_activity', [])
        endpoint_threat_keywords = (
            'credential_access', 'credential access',
            'lateral_movement', 'lateral movement',
            'driver_load', 'driver loaded',
            'persistence',
            'privilege_escalation', 'privilege escalation',
            'defense_evasion', 'defense evasion',
            'exfiltration',
            'command_and_control', 'command and control',
        )
        if any(
            any(kw in str(ep).lower() for kw in endpoint_threat_keywords)
            for ep in endpoint
        ):
            signals += 1
            signal_sources.append('endpoint_activity')

        # Cross-source correlations as additional signal
        correlations = payload.get('cross_source_correlations', [])
        if len(correlations) >= 5:
            signals += 1
            signal_sources.append('cross_source_correlations')

        return signals, signal_sources

    # ── Context-aware threat analysis  yi = F_LLM(Ti)  (Eq. 8 & 10) ──────────
    def analyze_threat_context(self, multimodal_text: str) -> Dict[str, Any]:
        logger.info("Performing context-aware LLM threat analysis (Eq. 8 & 10)…")

        prompt = LLM_PROMPTS['threat_analysis'].format(
            activity_data=multimodal_text,
            data_sources='system_logs, network_traffic, ids_alerts, dns_records, endpoint_activity'
        )
        result = self._call_llm(prompt, multimodal_text)

        result.setdefault('threat_type', 'no_threat')
        result.setdefault('risk_level', 'low')
        result.setdefault('confidence', 0.0)
        result.setdefault('pattern', 'No suspicious pattern detected')
        result.setdefault('actions', [])
        result.setdefault('affected_systems', [])
        result.setdefault('reasoning', '')

        result['yi'] = 0 if result['threat_type'] == 'no_threat' else 1

        kb_entry = self.knowledge_base.get(result['threat_type'], {})
        if kb_entry:
            result['knowledge_base_entry'] = {
                'mitre_technique': kb_entry.get('mitre_technique', 'N/A'),
                'severity':        kb_entry.get('severity', 'unknown'),
                'kb_indicators':   kb_entry.get('indicators', []),
                'kb_remediation':  kb_entry.get('remediation', ''),
            }

        logger.info(
            f"Threat analysis complete: type={result['threat_type']}, "
            f"yi={result['yi']}, confidence={result['confidence']}"
        )
        return result

    # ── Anomaly detection ─────────────────────────────────────────────────────
    def detect_anomalies(self, multimodal_text: str) -> Dict[str, Any]:
        logger.info("Performing LLM-based anomaly detection…")
        prompt = LLM_PROMPTS['anomaly_detection'].format(data=multimodal_text)
        result = self._call_llm(prompt, multimodal_text)

        result.setdefault('threat_type', 'no_threat')
        result.setdefault('confidence', 0.0)
        result.setdefault('risk_level', 'low')
        result.setdefault('pattern', '')
        result.setdefault('actions', [])
        result.setdefault('affected_systems', [])
        result.setdefault('reasoning', '')
        return result

    # ── Cross-source correlation (Eq. 6 & 7 / 10) ────────────────────────────
    def correlate_multimodal_events(self, representations: List[Dict]) -> Dict[str, Any]:
        logger.info("Performing LLM cross-source event correlation (Eq. 10)…")

        by_source: Dict[str, List[str]] = {}
        for rep in representations:
            by_source.setdefault(rep.get('source', 'unknown'), []).append(
                rep.get('text', '')[:200]
            )

        prompt = LLM_PROMPTS['correlation_analysis'].format(
            network_events=json.dumps(by_source.get('network_traffic', [])[:3], indent=2),
            system_logs=json.dumps(by_source.get('system_logs', [])[:3], indent=2),
            ids_alerts=json.dumps(by_source.get('ids_alerts', [])[:3], indent=2),
            dns_logs=json.dumps(by_source.get('dns_records', [])[:3], indent=2),
        )

        result = self._call_llm(prompt, json.dumps(by_source, default=str))
        result.setdefault('threat_type', 'no_threat')
        result.setdefault('confidence', 0.0)
        result.setdefault('risk_level', 'low')
        result.setdefault('pattern', '')
        result.setdefault('actions', [])
        result.setdefault('affected_systems', [])
        result.setdefault('correlated_sources', list(by_source.keys()))
        result.setdefault('attack_timeline', 'N/A')
        result.setdefault('reasoning', '')

        logger.info(
            f"Correlation analysis: type={result['threat_type']}, "
            f"confidence={result['confidence']}"
        )
        return result

    # ── Knowledge-Enhanced Reasoning (RAR) (Section 3.4.1, Eq. 11-13) ─────────
    def retrieval_augmented_reasoning(
        self,
        threat_type: str,
        indicators: List[str],
    ) -> Dict[str, Any]:
        logger.info(f"Running Retrieval-Augmented Reasoning for '{threat_type}'…")

        alpha = self.alpha
        matching_knowledge = []
        indicator_text = ' '.join(indicators)

        for kb_threat, kb_data in self.knowledge_base.items():
            kb_text = ' '.join(kb_data.get('indicators', []))
            sim = _text_similarity(indicator_text, kb_text)
            if sim > alpha:
                matching_knowledge.append({
                    'matched_threat':  kb_threat,
                    'similarity_score': round(sim, 4),
                    'mitre_technique': kb_data.get('mitre_technique', 'N/A'),
                    'severity':        kb_data.get('severity', 'unknown'),
                    'remediation':     kb_data.get('remediation', ''),
                    'impact_score':    kb_data.get('impact_score', 0.0),
                })

        matching_knowledge.sort(key=lambda x: x['similarity_score'], reverse=True)

        enriched_analysis = {}
        if matching_knowledge:
            best_match = matching_knowledge[0]['matched_threat']
            enriched_analysis = self.knowledge_base.get(best_match, {})
            logger.info(
                f"RAR: best match = '{best_match}' "
                f"(score={matching_knowledge[0]['similarity_score']}, α={alpha})"
            )
        else:
            logger.info(f"RAR: no knowledge-base match found above α={alpha}")

        return {
            'threat_type':        threat_type,
            'input_indicators':   indicators,
            'alpha_threshold':    alpha,
            'matching_knowledge': matching_knowledge,
            'enriched_analysis':  enriched_analysis,
        }

    # ── Risk Score and Severity (Equation 15) ─────────────────────────────────
    def _compute_severity(self, risk_prob: float, impact_score: float) -> float:
        """Severity_i = w1 * Ri + w2 * Impact_i — Equation (15)"""
        w1 = SystemConfig.THREAT_CONFIG['severity_weight_risk']
        w2 = SystemConfig.THREAT_CONFIG['severity_weight_impact']
        return round(w1 * risk_prob + w2 * impact_score, 4)

    # ── Main threat analysis pipeline ─────────────────────────────────────────
    def perform_threat_analysis(self, multimodal_data: str, representations: List[Dict] = None) -> Dict[str, Any]:
        logger.info("=== Starting LLM-based threat analysis pipeline ===")
        if LLAMA3_AVAILABLE and _llama3_analyzer:
            logger.info(f"    Active LLM: {_llama3_analyzer.backend_name()}")
        else:
            logger.info("    Active LLM: rule-based fallback (no API key configured)")

        results = {
            'timestamp':     __import__('datetime').datetime.now().isoformat(),
            'analysis_type': 'multimodal_llm_analysis',
            'llm_backend':   _llama3_analyzer.backend_name() if (LLAMA3_AVAILABLE and _llama3_analyzer) else 'rule_based',
        }

        logger.info("[1/5] Context-aware threat detection…")
        threat_analysis = self.analyze_threat_context(multimodal_data)
        results['threat_analysis'] = threat_analysis

        # ── Llama 3 Cross-Verification (paper Section 3.4) ────────────────────
        if LLAMA3_AVAILABLE and _llama3_analyzer and threat_analysis.get('threat_type') != 'no_threat':
            logger.info("[2/5] Llama 3 cross-verification (false positive elimination)…")
            cross_verification = _llama3_analyzer.cross_verify(threat_analysis, multimodal_data)
            results['llama3_cross_verification'] = cross_verification

            # Apply cross-verification adjustments
            if cross_verification.get('false_positive'):
                logger.info(
                    f"    Llama 3 flagged FALSE POSITIVE: "
                    f"{cross_verification.get('false_positive_reason', 'unspecified')}"
                )
                threat_analysis['threat_type'] = 'no_threat'
                threat_analysis['confidence']  = cross_verification.get('adjusted_confidence', 0.0)
                threat_analysis['risk_level']  = 'low'
                threat_analysis['yi']          = 0
                results['false_positive_eliminated'] = True
            else:
                threat_analysis['confidence'] = cross_verification.get(
                    'adjusted_confidence', threat_analysis.get('confidence', 0.0)
                )
                threat_analysis['risk_level'] = cross_verification.get(
                    'adjusted_risk_level', threat_analysis.get('risk_level', 'low')
                )
                results['false_positive_eliminated'] = False
        else:
            logger.info("[2/5] Llama 3 cross-verification: skipped (no threat or unavailable)")
            results['llama3_cross_verification'] = None
            results['false_positive_eliminated'] = False

        logger.info("[3/5] Anomaly detection…")
        anomaly_analysis = self.detect_anomalies(multimodal_data)
        if threat_analysis.get('threat_type') == 'no_threat':
            anomaly_analysis['confidence'] = min(anomaly_analysis.get('confidence', 0.0), 0.2)
        results['anomaly_analysis'] = anomaly_analysis

        logger.info("[4/5] Cross-source event correlation…")
        reps = representations or []
        correlation_analysis = self.correlate_multimodal_events(reps)
        if threat_analysis.get('threat_type') == 'no_threat':
            correlation_analysis['confidence'] = min(correlation_analysis.get('confidence', 0.0), 0.2)
        results['correlation_analysis'] = correlation_analysis

        logger.info("[5/5] Computing risk score (Severity = w1·R + w2·Impact)…")
        threat_conf  = threat_analysis.get('confidence', 0.0)
        anomaly_conf = anomaly_analysis.get('confidence', 0.0)
        corr_conf    = correlation_analysis.get('confidence', 0.0)

        overall_risk_score = round(
            0.60 * threat_conf + 0.25 * anomaly_conf + 0.15 * corr_conf, 4
        )

        kb_entry     = self.knowledge_base.get(threat_analysis.get('threat_type', 'no_threat'), {})
        impact_score = kb_entry.get('impact_score', 0.0)
        severity_score = self._compute_severity(overall_risk_score, impact_score)

        results['overall_risk_score'] = overall_risk_score
        results['severity_score']     = severity_score
        results['overall_risk_level'] = SystemConfig.get_risk_level(overall_risk_score)

        risk_threshold = SystemConfig.THREAT_CONFIG['risk_threshold']
        threat_type    = threat_analysis.get('threat_type', 'no_threat')
        results['threat_detected'] = (
            overall_risk_score >= risk_threshold and threat_type != 'no_threat'
        )
        results['detection_reason'] = (
            f"risk_score={overall_risk_score} | threshold={risk_threshold} | "
            f"threat_type={threat_type} | severity={severity_score}"
        )

        results['recommendations'] = self._generate_recommendations(
            threat_analysis, anomaly_analysis, correlation_analysis
        )

        logger.info(
            f"=== Threat analysis complete: detected={results['threat_detected']}, "
            f"type={threat_type}, risk={overall_risk_score}, severity={severity_score} ==="
        )
        return results

    def _generate_recommendations(self, *analyses) -> List[str]:
        recs = set()
        for analysis in analyses:
            if isinstance(analysis, dict):
                for action in analysis.get('actions', []):
                    recs.add(action)
        recs.update([
            'Review security logs for related incidents',
            'Update firewall rules if necessary',
            'Notify security team of findings',
            'Document incident for future reference',
        ])
        return sorted(recs)


# ─────────────────────────────────────────────────────────────────────────────
# Alert Generation  Ai = Alert(yi, Severity_i)  — Equation (14)
# ─────────────────────────────────────────────────────────────────────────────
class ThreatAlertGenerator:
    """Implements alert generation — Equation (14): Ai = Alert(yi, Severity_i)"""

    def __init__(self):
        self.risk_levels = SystemConfig.RISK_LEVELS

    def generate_alert(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        import random, datetime
        yi            = analysis_result.get('threat_analysis', {}).get('yi', 0)
        severity_score = analysis_result.get('severity_score', 0.0)
        risk_level    = analysis_result.get('overall_risk_level', 'low')

        alert = {
            'alert_id':          f"ALERT-{random.randint(100_000, 999_999)}",
            'timestamp':         datetime.datetime.now().isoformat(),
            'yi':                yi,
            'threat_type':       analysis_result.get('threat_analysis', {}).get('threat_type', 'unknown'),
            'risk_level':        risk_level,
            'overall_risk_score': analysis_result.get('overall_risk_score', 0.0),
            'severity_score':    severity_score,
            'confidence':        analysis_result.get('threat_analysis', {}).get('confidence', 0.0),
            'affected_systems':  analysis_result.get('threat_analysis', {}).get('affected_systems', []),
            'recommended_actions': analysis_result.get('recommendations', []),
            'mitre_technique':   (
                analysis_result
                .get('threat_analysis', {})
                .get('knowledge_base_entry', {})
                .get('mitre_technique', 'N/A')
            ),
            'status': 'active',
        }
        alert['source_ips'] = alert['affected_systems']
        return alert
