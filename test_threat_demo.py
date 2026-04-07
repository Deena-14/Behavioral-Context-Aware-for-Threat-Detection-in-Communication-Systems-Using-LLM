"""
Threat Detection DEMO script
Forces high-suspicion multimodal data so that a threat is always detected.
Run: python test_threat_demo.py
"""

import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import SystemConfig, THREAT_KNOWLEDGE_BASE
from llm_analysis import LLMThreatAnalyzer, ThreatAlertGenerator
from threat_alert import ThreatAlertManager, AlertNotifier

# ── Force a high-threat multimodal input ──────────────────────────────────────
# This simulates what the system sees when a REAL attack is happening
FORCED_THREAT_INPUT = json.dumps({
    "analysis_timestamp": datetime.now().isoformat(),
    "data_sources_integrated": [
        "system_logs", "network_traffic", "ids_alerts", "dns_records", "endpoint_activity"
    ],
    "system_logs": [
        "[SYSTEM LOG] Event: PRIVILEGE_ESCALATION | User: user7 | Host: host-12 | IP: 192.168.1.45 | Severity: critical | Time: 2026-03-30T13:00:00",
        "[SYSTEM LOG] Event: AUTH_FAILURE | User: admin | Host: host-3 | IP: 192.168.1.102 | Severity: high | Time: 2026-03-30T13:01:00",
        "[SYSTEM LOG] Event: FAILED_LOGIN | User: root | Host: host-22 | IP: 192.168.1.200 | Severity: high | Time: 2026-03-30T13:01:30",
    ],
    "network_traffic": [
        "[NETWORK TRAFFIC] 192.168.1.45:54321 → 185.220.101.5:4444 | Proto: TCP | Bytes: 4500000 | Packets: 9200 | Flags: SYN | Time: 2026-03-30T13:02:00",
        "[NETWORK TRAFFIC] 192.168.1.102:61234 → 91.108.4.1:31337 | Proto: TCP | Bytes: 2300000 | Packets: 4500 | Flags: ACK | Time: 2026-03-30T13:03:00",
    ],
    "ids_alerts": [
        "[IDS ALERT] Buffer Overflow Attempt | Src: 192.168.1.45 → Dst: 10.0.0.5 | Sig: SID-7823451 | Severity: critical | Time: 2026-03-30T13:02:30",
        "[IDS ALERT] Malware Signature Match | Src: 192.168.1.102 → Dst: 10.0.0.8 | Sig: SID-9912345 | Severity: critical | Time: 2026-03-30T13:03:30",
    ],
    "dns_records": [
        "[DNS RECORD] Query from 192.168.1.45 for botnet-c2.ru (A) | Response: NOERROR | Count: 87 | Suspicious: True | Time: 2026-03-30T13:04:00",
        "[DNS RECORD] Query from 192.168.1.102 for dga-abc123xyz.io (TXT) | Response: NOERROR | Count: 143 | Suspicious: True | Time: 2026-03-30T13:04:30",
    ],
    "endpoint_activity": [
        "[ENDPOINT] Credential Access on endpoint-45 | User: user7 | Process: powershell.exe (PID 4821) | Severity: critical | Time: 2026-03-30T13:05:00",
        "[ENDPOINT] Driver Loaded on endpoint-12 | User: admin | Process: cmd.exe (PID 1234) | Severity: critical | Time: 2026-03-30T13:05:30",
    ],
    "cross_source_correlations": [
        "Cross-source correlation: 192.168.1.45 appears in system_logs + network_traffic + ids_alerts + dns_records",
        "Cross-source correlation: 192.168.1.102 appears in system_logs + network_traffic + dns_records + endpoint_activity",
        "Cross-source correlation: PRIVILEGE_ESCALATION → Buffer Overflow → C2 beacon → Credential Access (APT kill chain)",
        "Cross-source correlation: Lateral movement detected across host-12, host-3, host-22",
    ]
}, indent=2)


def run_threat_demo():
    print("\n" + "=" * 70)
    print("   🔴  THREAT DETECTION DEMO — FORCED HIGH-RISK SCENARIO")
    print("=" * 70)
    print("\n📡 Multimodal data injected from 5 sources:")
    print("   ✅ System Logs      → PRIVILEGE_ESCALATION, AUTH_FAILURE, FAILED_LOGIN")
    print("   ✅ Network Traffic  → C2 beacon to 185.220.101.5:4444 (known bad IP)")
    print("   ✅ IDS Alerts       → Buffer Overflow + Malware Signature (CRITICAL)")
    print("   ✅ DNS Records      → botnet-c2.ru, dga-abc123xyz.io (suspicious)")
    print("   ✅ Endpoint Activity→ Credential Access + Driver Load (CRITICAL)")
    print("\n" + "-" * 70)

    config = SystemConfig.to_dict()
    analyzer     = LLMThreatAnalyzer(config['llm'])
    alert_gen    = ThreatAlertGenerator()
    alert_manager= ThreatAlertManager()
    notifier     = AlertNotifier(config['alert'])

    # ── Step 1: LLM Threat Analysis ───────────────────────────────────────────
    print("\n[STEP 1] 🧠 LLM-Based Threat Analysis (Equations 8-15)...")
    analysis = analyzer.perform_threat_analysis(FORCED_THREAT_INPUT, representations=[])

    ta = analysis['threat_analysis']
    print(f"\n   Threat Type    : {ta['threat_type'].upper()}")
    print(f"   yi (Eq.9)      : {ta.get('yi')}  → {'🚨 THREAT' if ta.get('yi') == 1 else '✅ NORMAL'}")
    print(f"   Risk Level     : {analysis['overall_risk_level'].upper()}")
    print(f"   Risk Score     : {analysis['overall_risk_score']} (threshold: 0.55)")
    print(f"   Severity       : {analysis['severity_score']}  (Eq.15: w1·R + w2·Impact)")
    print(f"   Confidence     : {ta['confidence']}")
    print(f"   Threat Detected: {'🔴 YES' if analysis['threat_detected'] else '🟢 NO'}")

    print(f"\n   📋 Pattern Detected:")
    print(f"   {ta['pattern']}")

    print(f"\n   🔍 Reasoning:")
    print(f"   {ta['reasoning']}")

    print(f"\n   🖥️  Affected Systems: {ta.get('affected_systems', [])}")

    if 'knowledge_base_entry' in ta:
        kb = ta['knowledge_base_entry']
        print(f"\n   📚 MITRE ATT&CK : {kb.get('mitre_technique')}")
        print(f"   KB Severity     : {kb.get('severity')}")
        print(f"   KB Remediation  : {kb.get('kb_remediation')}")

    # ── Step 2: Retrieval-Augmented Reasoning ─────────────────────────────────
    print("\n" + "-" * 70)
    print("\n[STEP 2] 📖 Retrieval-Augmented Reasoning (Equations 11-13)...")
    indicators = ['lateral_movement', 'command_and_control', 'privilege_escalation',
                  'persistence_mechanism', 'credential_access']
    rar = analyzer.retrieval_augmented_reasoning(ta['threat_type'], indicators)

    print(f"\n   α threshold     : {rar['alpha_threshold']}")
    print(f"   KB Matches Found: {len(rar['matching_knowledge'])}")
    for match in rar['matching_knowledge'][:3]:
        print(f"   → Matched '{match['matched_threat']}' "
              f"(sim={match['similarity_score']}, MITRE={match['mitre_technique']})")
    if rar['enriched_analysis']:
        print(f"   Remediation     : {rar['enriched_analysis'].get('remediation','N/A')}")

    # ── Step 3: Alert Generation ──────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("\n[STEP 3] 🚨 Alert Generation — Ai = Alert(yi, Severity_i) [Eq. 14]...")
    alert = alert_gen.generate_alert(analysis)
    alert = alert_manager.create_alert(analysis)
    notifier.broadcast_alert(alert)

    print(f"\n   ┌─────────────────────────────────────────────────┐")
    print(f"   │  🚨 SECURITY ALERT GENERATED                    │")
    print(f"   ├─────────────────────────────────────────────────┤")
    print(f"   │  Alert ID    : {alert['alert_id']:<33} │")
    print(f"   │  Threat      : {alert['threat_type'].upper():<33} │")
    print(f"   │  Risk Level  : {alert['risk_level'].upper():<33} │")
    print(f"   │  Risk Score  : {str(alert['risk_score']):<33} │")
    print(f"   │  Timestamp   : {str(alert['timestamp'])[:33]:<33} │")
    print(f"   │  Investigate : {'YES ⚠️' if alert['investigation_required'] else 'NO':<33} │")
    print(f"   └─────────────────────────────────────────────────┘")

    print(f"\n   📌 Recommended Actions:")
    for i, action in enumerate(alert['actions_recommended'], 1):
        print(f"   {i}. {action}")

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    summary = alert_manager.generate_alert_summary()
    print(f"\n[STEP 4] 📊 Alert Summary")
    print(f"   Total Alerts   : {summary['total_alerts']}")
    print(f"   Critical Alerts: {summary['critical_alerts']}")
    print(f"   High Risk      : {summary['high_risk_alerts']}")

    print("\n" + "=" * 70)
    print("   ✅ DEMO COMPLETE — Threat successfully detected & alerted!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_threat_demo()
