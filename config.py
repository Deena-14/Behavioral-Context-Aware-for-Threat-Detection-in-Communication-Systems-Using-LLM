"""
Configuration module for Multimodal LLM-based Cybersecurity System
Aligned with: "AI Driven LLM Enhanced Multimodal Cybersecurity Threat Detection
               in Communication Networks" (Taylor et al.)

CHANGES (v2):
  - Switched LLM provider from OpenAI → Anthropic (claude-haiku-4-5-20251001)
  - dataset_path now reads from KDD_DATASET_PATH env variable (configurable)
  - test_dataset_path added for KDDTest+.arff evaluation support
  - Removed dummy 'sk-test-key' fallback; empty string handled explicitly in llm_analysis.py
"""

import os
from typing import Dict, Any


class SystemConfig:
    """System configuration settings"""

    # ── Llama 3 config (primary LLM — paper's described model) ──────────────
    # Option A: Groq API (free, cloud) — https://console.groq.com
    #   Set env: GROQ_API_KEY=<your-key>
    # Option B: Ollama (fully local, no API key)
    #   Install: https://ollama.com  then run: ollama pull llama3
    LLAMA3_CONFIG = {
        'provider':      'groq',
        'groq_model':    'llama3-70b-8192',
        'ollama_model':  'llama3',
        'ollama_host':   os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
        'groq_api_key':  os.getenv('GROQ_API_KEY', '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'),
        'temperature':   0.1,
        'max_tokens':    1024,
    }

    # ── Anthropic Claude config (secondary fallback LLM) ─────────────────────
    LLM_CONFIG = {
        'provider':    'anthropic',
        'model_name':  'claude-haiku-4-5-20251001',
        'temperature': 0.3,
        'max_tokens':  1024,
        'api_key':     os.getenv('ANTHROPIC_API_KEY', ''),
    }

    NETWORK_CONFIG = {
        'capture_interface': None,
        'packet_count': 100,
        'filter': 'ip',
        'timeout': 60,
    }

    DATA_SOURCES = {
        'network_traffic': True,
        'system_logs': True,
        'ids_alerts': True,
        'dns_records': True,
        'endpoint_activity': True,
    }

    THREAT_CONFIG = {
        'enabled_sources': ['network_traffic', 'system_logs', 'ids_alerts', 'dns_records', 'endpoint_activity'],
        'correlation_window': 300,
        'similarity_threshold': 0.7,
        'knowledge_match_threshold': 0.15,
        'risk_threshold': 0.50,
        'alert_cooldown': 60,
        'severity_weight_risk': 0.6,
        'severity_weight_impact': 0.4,
    }

    PROCESSING_CONFIG = {
        'normalize_data': True,
        'extract_features': True,
        'correlate_events': True,
        'aggregate_windows': 10,
        'embedding_dim': 768,
        # CHANGE: Both paths now env-configurable
        'dataset_path':      os.getenv('KDD_DATASET_PATH', 'KDDTrain+.txt'),
        'test_dataset_path': os.getenv('KDD_TEST_PATH',    'KDDTest+.arff'),
        'dataset_name': 'NSL-KDD',
    }

    ALERT_CONFIG = {
        'alert_types': ['email', 'slack', 'log_file', 'dashboard'],
        'log_file': 'logs/alerts.log',
        'email_enabled': False,
        'slack_enabled': False,
        'dashboard_port': 8501,
    }

    DATABASE_CONFIG = {
        'type': 'sqlite',
        'path': 'data/threats.db',
        'cache_size': 1000,
    }

    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'logs/system.log',
    }

    RISK_LEVELS = {
        'critical': {'score': (0.9, 1.0), 'color': '#FF0000'},
        'high':     {'score': (0.7, 0.9), 'color': '#FF6600'},
        'medium':   {'score': (0.4, 0.7), 'color': '#FFFF00'},
        'low':      {'score': (0.0, 0.4), 'color': '#00FF00'},
    }

    THREAT_CATEGORIES = [
        'credential_stuffing', 'brute_force', 'data_exfiltration',
        'ddos_attack', 'insider_threat', 'apt_activity', 'malware_detected',
        'suspicious_dns', 'port_scanning', 'privilege_escalation',
        'ransomware', 'no_threat',
    ]

    @classmethod
    def get_risk_level(cls, score: float) -> str:
        for level, cfg in cls.RISK_LEVELS.items():
            if cfg['score'][0] <= score < cfg['score'][1]:
                return level
        return 'critical' if score >= 0.9 else 'low'

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            'llm':        cls.LLM_CONFIG,
            'network':    cls.NETWORK_CONFIG,
            'sources':    cls.DATA_SOURCES,
            'threat':     cls.THREAT_CONFIG,
            'processing': cls.PROCESSING_CONFIG,
            'alert':      cls.ALERT_CONFIG,
            'database':   cls.DATABASE_CONFIG,
            'logging':    cls.LOGGING_CONFIG,
        }


# ── Knowledge Base  K = {k1, k2, …, kp}  (Equation 11) ──────────────────────
THREAT_KNOWLEDGE_BASE = {
    'credential_stuffing': {
        'indicators': ['multiple_login_attempts', 'invalid_credentials', 'same_ip',
                       'rate_anomaly', 'user_enumeration'],
        'severity': 'high',
        'mitre_technique': 'T1110.004',
        'remediation': 'Enable MFA, Block source IP, Reset credentials, Rate-limit auth endpoints',
        'impact_score': 0.75,
    },
    'brute_force': {
        'indicators': ['rapid_attempts', 'failed_auth', 'sequential_passwords',
                       'dictionary_attack', 'port_scanning'],
        'severity': 'high',
        'mitre_technique': 'T1110',
        'remediation': 'Apply rate limiting, Block IP, Enforce strong passwords, Alert SOC',
        'impact_score': 0.70,
    },
    'data_exfiltration': {
        'indicators': ['large_transfer', 'unusual_dest_ip', 'admin_access',
                       'suspicious_dns', 'off_hours_upload', 'compressed_outbound'],
        'severity': 'critical',
        'mitre_technique': 'T1041',
        'remediation': 'Block outbound connection, Isolate system, Preserve forensic evidence, Notify CISO',
        'impact_score': 0.95,
    },
    'ddos_attack': {
        'indicators': ['traffic_spike', 'many_source_ips', 'slow_response',
                       'syn_flood', 'udp_flood', 'amplification'],
        'severity': 'high',
        'mitre_technique': 'T1498',
        'remediation': 'Activate DDoS mitigation, Apply geo-blocking, Rate-limit, Contact ISP',
        'impact_score': 0.80,
    },
    'insider_threat': {
        'indicators': ['off_hours_access', 'sensitive_file_access', 'unauthorized_export',
                       'privilege_abuse', 'bulk_download'],
        'severity': 'critical',
        'mitre_technique': 'T1078',
        'remediation': 'Revoke user access, Audit activity, Engage HR and legal',
        'impact_score': 0.90,
    },
    'apt_activity': {
        'indicators': ['persistence_mechanism', 'lateral_movement', 'command_and_control',
                       'living_off_land', 'spearphishing', 'zero_day_exploit'],
        'severity': 'critical',
        'mitre_technique': 'TA0001-TA0011',
        'remediation': 'Full incident response, Endpoint isolation, Memory forensics, Threat hunt',
        'impact_score': 0.98,
    },
    'malware_detected': {
        'indicators': ['malware_signature', 'suspicious_process', 'encrypted_c2_traffic',
                       'registry_modification', 'dll_injection'],
        'severity': 'high',
        'mitre_technique': 'T1059',
        'remediation': 'Quarantine endpoint, Full AV scan, Restore from backup, Patch vulnerabilities',
        'impact_score': 0.82,
    },
    'suspicious_dns': {
        'indicators': ['dns_tunneling', 'high_query_rate', 'nxdomain_storm',
                       'dga_domain', 'long_subdomain', 'rare_tld'],
        'severity': 'medium',
        'mitre_technique': 'T1071.004',
        'remediation': 'Block suspicious domains, Enable DNS filtering, Inspect for tunneling',
        'impact_score': 0.60,
    },
    'port_scanning': {
        'indicators': ['sequential_port_access', 'rapid_connection_attempts',
                       'syn_without_ack', 'half_open_connections'],
        'severity': 'medium',
        'mitre_technique': 'T1046',
        'remediation': 'Block source IP, Enable port-knocking, Review firewall rules',
        'impact_score': 0.55,
    },
    'privilege_escalation': {
        'indicators': ['sudo_abuse', 'token_manipulation', 'setuid_exploitation',
                       'kernel_exploit', 'bypass_uac'],
        'severity': 'high',
        'mitre_technique': 'T1068',
        'remediation': 'Revoke elevated privileges, Patch OS, Review sudoers, Enable PAM',
        'impact_score': 0.85,
    },
    'ransomware': {
        'indicators': ['mass_file_encryption', 'ransom_note_creation', 'shadow_copy_deletion',
                       'c2_beacon', 'lateral_movement'],
        'severity': 'critical',
        'mitre_technique': 'T1486',
        'remediation': 'Isolate systems immediately, Restore from offline backups, Engage IR team',
        'impact_score': 0.99,
    },
    'no_threat': {
        'indicators': [],
        'severity': 'none',
        'mitre_technique': 'N/A',
        'remediation': 'Continue monitoring — no action required',
        'impact_score': 0.0,
    },
}


# ── LLM Prompts aligned with paper's semantic reasoning (Section 3.4) ─────────
LLM_PROMPTS = {
    'threat_analysis': """
You are a cybersecurity AI performing multimodal threat detection for a communication network.

MULTIMODAL SECURITY DATA (from 5 integrated sources):
{activity_data}

DATA SOURCES PRESENT: {data_sources}

Perform context-aware semantic reasoning across ALL data sources. Identify:
1. THREAT TYPE — one of: credential_stuffing, brute_force, data_exfiltration,
   ddos_attack, insider_threat, apt_activity, malware_detected, suspicious_dns,
   port_scanning, privilege_escalation, ransomware, no_threat
2. RISK LEVEL — critical / high / medium / low
3. CONFIDENCE SCORE — float 0.0 to 1.0
4. ATTACK PATTERN — description of what you observed across modalities
5. RECOMMENDED ACTIONS — list of strings
6. AFFECTED SYSTEMS — list of IPs or hostnames

Respond ONLY with valid JSON (no extra text):
{{
  "threat_type": "...",
  "risk_level": "...",
  "confidence": 0.0,
  "pattern": "...",
  "actions": ["..."],
  "affected_systems": ["..."],
  "reasoning": "..."
}}
""",

    'correlation_analysis': """
You are a cybersecurity AI performing cross-source event correlation (multimodal analysis).

NETWORK EVENTS:
{network_events}

SYSTEM LOGS:
{system_logs}

IDS ALERTS:
{ids_alerts}

DNS LOGS:
{dns_logs}

Identify correlations between events. Look for:
- Same IP across multiple sources
- Temporal clusters of suspicious events
- Multi-stage attack chains (recon -> exploit -> exfiltration)

Respond ONLY with valid JSON:
{{
  "threat_type": "...",
  "confidence": 0.0,
  "risk_level": "...",
  "pattern": "...",
  "actions": ["..."],
  "affected_systems": ["..."],
  "correlated_sources": ["..."],
  "attack_timeline": "...",
  "reasoning": "..."
}}
""",

    'anomaly_detection': """
You are a cybersecurity AI detecting anomalies in multimodal network security data.

DATA:
{data}

Flag deviations from normal baseline:
- Unusual access patterns or times
- Abnormal traffic volumes or destinations
- Unexpected process execution
- Suspicious DNS queries
- Privilege escalation attempts

Respond ONLY with valid JSON:
{{
  "threat_type": "...",
  "confidence": 0.0,
  "risk_level": "...",
  "pattern": "...",
  "actions": ["..."],
  "affected_systems": ["..."],
  "reasoning": "..."
}}
""",
}
