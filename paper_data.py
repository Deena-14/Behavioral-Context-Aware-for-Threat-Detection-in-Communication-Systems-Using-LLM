"""
paper_data.py
All ground-truth data, tables, and results directly from:
  "AI Driven LLM Enhanced Multimodal Cybersecurity Threat Detection
   in Communication Networks" — Taylor et al.

Every number here is taken verbatim from the paper so the
simulation and dashboard always reflect the published results.
"""

# ── Table 2: Benchmark Datasets (Section 4.1) ────────────────────────────────
BENCHMARK_DATASETS = [
    {
        "name": "CICIDS2017",
        "data_type": "Network Traffic",
        "records": "2.8 Million",
        "attack_types": "DDoS, Brute Force, Botnet",
        "purpose": "Intrusion detection evaluation",
    },
    {
        "name": "UNSW-NB15",
        "data_type": "Network Flow",
        "records": "2.5 Million",
        "attack_types": "DoS, Exploits, Reconnaissance",
        "purpose": "Modern network attack detection",
    },
    {
        "name": "NSL-KDD",
        "data_type": "Network Traffic",
        "records": "148,000",
        "attack_types": "Probe, DoS, R2L, U2R",
        "purpose": "Benchmark IDS evaluation",
    },
    {
        "name": "DARPA 1999",
        "data_type": "Network & System Logs",
        "records": "4.9 Million",
        "attack_types": "DoS, Probing, Data Theft",
        "purpose": "Early intrusion detection benchmark",
    },
    {
        "name": "Bot-IoT",
        "data_type": "IoT Network Traffic",
        "records": "72 Million",
        "attack_types": "Botnet, DDoS, Data Exfiltration",
        "purpose": "IoT attack detection",
    },
]

# ── Table 3 / Table 6: Model Performance Comparison (Section 5.1 & 5.4) ──────
MODEL_COMPARISON = [
    {"model": "Random Forest",          "accuracy": 88.4, "precision": 87.2, "recall": 86.5, "f1": 86.8},
    {"model": "Support Vector Machine", "accuracy": 89.6, "precision": 88.9, "recall": 87.7, "f1": 88.3},
    {"model": "Deep Neural Network",    "accuracy": 91.2, "precision": 90.4, "recall": 90.1, "f1": 90.2},
    {"model": "Proposed Model",         "accuracy": 93.2, "precision": 92.6, "recall": 92.0, "f1": 92.3},
]

# ── Table 4: Detection Performance per Attack Type (Section 5.2) ─────────────
ATTACK_TYPE_PERFORMANCE = [
    {"attack": "DoS / DDoS",     "precision": 94.1, "recall": 93.5, "f1": 93.8},
    {"attack": "Brute Force",    "precision": 92.7, "recall": 91.9, "f1": 92.3},
    {"attack": "Botnet",         "precision": 93.4, "recall": 92.8, "f1": 93.1},
    {"attack": "Insider Attack", "precision": 91.5, "recall": 90.8, "f1": 91.1},
]

# ── Table 5: False Alarm Rate Comparison (Section 5.3) ───────────────────────
FALSE_ALARM_COMPARISON = [
    {"method": "Signature Based IDS",  "detection_accuracy": 82.5, "false_alarm_rate": 18.3},
    {"method": "Rule Based Detection", "detection_accuracy": 85.1, "false_alarm_rate": 15.9},
    {"method": "Machine Learning IDS", "detection_accuracy": 90.4, "false_alarm_rate": 11.2},
    {"method": "Proposed Model",       "detection_accuracy": 93.2, "false_alarm_rate":  9.5},
]

# ── Table 7: Attack Category Performance (Section 5.5) ───────────────────────
ATTACK_CATEGORY_PERFORMANCE = [
    {"attack": "DoS / DDoS", "accuracy": 94.1, "precision": 93.5, "recall": 93.0, "f1": 93.2},
    {"attack": "Phishing",   "accuracy": 92.8, "precision": 92.1, "recall": 91.6, "f1": 91.8},
    {"attack": "Malware",    "accuracy": 93.4, "precision": 92.7, "recall": 92.2, "f1": 92.4},
    {"attack": "Botnet",     "accuracy": 91.9, "precision": 91.3, "recall": 90.8, "f1": 91.0},
    {"attack": "Average",    "accuracy": 93.2, "precision": 92.6, "recall": 92.0, "f1": 92.3},
]

# ── Table 8: Performance Stability — Mean ± Std (Section 5.6) ────────────────
STABILITY_METRICS = [
    {"model": "Random Forest",          "acc_mean": 88.4, "acc_std": 0.72, "prec_mean": 87.2, "prec_std": 0.68, "rec_mean": 86.5, "rec_std": 0.75, "f1_mean": 86.8, "f1_std": 0.70},
    {"model": "Support Vector Machine", "acc_mean": 89.6, "acc_std": 0.65, "prec_mean": 88.9, "prec_std": 0.60, "rec_mean": 87.7, "rec_std": 0.64, "f1_mean": 88.3, "f1_std": 0.62},
    {"model": "Deep Neural Network",    "acc_mean": 91.2, "acc_std": 0.58, "prec_mean": 90.4, "prec_std": 0.55, "rec_mean": 90.1, "rec_std": 0.59, "f1_mean": 90.2, "f1_std": 0.57},
    {"model": "Proposed Model",         "acc_mean": 93.2, "acc_std": 0.44, "prec_mean": 92.6, "prec_std": 0.41, "rec_mean": 92.0, "rec_std": 0.46, "f1_mean": 92.3, "f1_std": 0.43},
]

# ── Abstract-level headline metrics ──────────────────────────────────────────
HEADLINE_METRICS = {
    "threat_detection_accuracy_pct":     94.3,   # Abstract: "94.3% threat detection accuracy"
    "false_positive_reduction_pct":      93.8,   # Abstract: "93.8% reduction in false positive alerts"
    "alert_volume_reduction_pct":        49.6,   # Abstract: "49.6% reduction in overall alert volume"
    "true_positive_increase_pct":         8.3,   # Abstract: "8.3% increase in true positive detections"
    "proposed_model_accuracy":           93.2,   # Table 3
    "proposed_model_precision":          92.6,
    "proposed_model_recall":             92.0,
    "proposed_model_f1":                 92.3,
    "proposed_model_false_alarm_rate":    9.5,   # Table 5
}

# ── Data source event counts used in simulation (realistic, paper-aligned) ───
SIMULATION_CONFIG = {
    "system_logs_count":       15,
    "network_traffic_count":   20,
    "ids_alerts_count":        10,
    "dns_records_count":       15,
    "endpoint_activity_count": 12,
    "total_sources":            5,
}

# ── Threat type → MITRE mapping used in knowledge base (paper Section 3.4.1) ─
MITRE_MAPPING = {
    "apt_activity":          "TA0001–TA0011",
    "data_exfiltration":     "T1041",
    "insider_threat":        "T1078",
    "malware_detected":      "T1059",
    "brute_force":           "T1110",
    "credential_stuffing":   "T1110.004",
    "privilege_escalation":  "T1068",
    "suspicious_dns":        "T1071.004",
    "port_scanning":         "T1046",
    "ransomware":            "T1486",
    "ddos_attack":           "T1498",
}