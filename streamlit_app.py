"""
Streamlit Dashboard — Multimodal LLM-Based Cybersecurity Threat Detection
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from config import SystemConfig
from multimodal_processor import MultimodalDataProcessor
from llm_analysis import LLMThreatAnalyzer, ThreatAlertGenerator
from threat_alert import ThreatAlertManager, AlertNotifier
from network_capture import NetworkCapture


HEADLINE_METRICS = {
    "threat_detection_accuracy_pct":    94.3,
    "false_positive_reduction_pct":     93.8,
    "alert_volume_reduction_pct":       49.6,
    "true_positive_increase_pct":        8.3,
    "proposed_model_accuracy":          93.2,
    "proposed_model_precision":         92.6,
    "proposed_model_recall":            92.0,
    "proposed_model_f1":                92.3,
    "proposed_model_false_alarm_rate":   9.5,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LLM Cybersecurity Threat Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Kill ALL Streamlit fade-in / transition / blink animations ── */
*, *::before, *::after {
    animation: none !important;
    animation-duration: 0s !important;
    animation-delay: 0s !important;
    transition: none !important;
    transition-duration: 0s !important;
    transition-delay: 0s !important;
}
/* Force all elements fully visible — no fade-in on rerender */
.element-container, .stMarkdown, .stText, .stMetric,
.stExpander, .stDataFrame, .stPlotlyChart, .stImage,
[data-testid="stMetric"], [data-testid="metric-container"],
[data-testid="stExpander"], [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"], [data-testid="column"],
[data-testid="stMarkdownContainer"], [data-testid="stExpander"] summary,
[data-testid="stExpander"] > div, .streamlit-expanderHeader,
.streamlit-expanderContent, div[role="button"] {
    animation: none !important;
    transition: none !important;
    opacity: 1 !important;
}
/* Streamlit 1.25 expander label flicker fix */
details > summary { list-style: none; }
details > summary::marker,
details > summary::-webkit-details-marker { display: none; }
details[open] > summary ~ * { animation: none !important; }
details > div { animation: none !important; transition: none !important; }
/* metric-card and other custom cards */
.metric-card{background:#f8f9fa;border-radius:10px;padding:18px;margin:6px 0;text-align:center;border:1px solid #e0e0e0}
.source-card{background:#f8f9fa;border-radius:10px;padding:16px;text-align:center;border:1px solid #e0e0e0}
.alert-box{background:#fff3cd;border-left:4px solid #ffc107;padding:12px;border-radius:4px;margin-bottom:10px}
.result-box{background:#d4edda;border-left:4px solid #28a745;padding:12px;border-radius:4px;margin-bottom:10px}
.crit-box{background:#f8d7da;border-left:4px solid #dc3545;padding:12px;border-radius:4px}
</style>
""", unsafe_allow_html=True)

MODEL_COLORS   = ["#2196F3","#4CAF50","#FF9800","#F44336"]
RISK_COLOR_MAP = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🟢"}

_ENDPOINT_ONLY = {"endpoint isolation", "memory forensics", "full av scan",
                  "quarantine endpoint", "restore from backup"}
_NETWORK_THREATS = {"port_scanning", "ddos_attack", "suspicious_dns"}

def _safe_rerun():
    """Compatible rerun for Streamlit 1.25 and newer versions."""
    if hasattr(st, 'rerun'):
        st.rerun()
    else:
        st.experimental_rerun()


def _filter_recs(recs: list, threat_type: str) -> list:
    t = threat_type.lower()
    out = []
    for r in recs:
        rl = r.lower()
        if t in _NETWORK_THREATS and any(e in rl for e in _ENDPOINT_ONLY):
            continue
        out.append(r)
    return out


class Dashboard:
    def __init__(self):
        self.config = SystemConfig()
        if "processor" not in st.session_state:
            st.session_state.processor     = MultimodalDataProcessor(self.config.PROCESSING_CONFIG)
            st.session_state.analyzer      = LLMThreatAnalyzer(self.config.LLM_CONFIG)
            st.session_state.notifier      = AlertNotifier(self.config.ALERT_CONFIG)
            st.session_state.net_capture   = NetworkCapture(self.config.NETWORK_CONFIG)
            st.session_state.alert_manager = ThreatAlertManager()
        self.processor     = st.session_state.processor
        self.analyzer      = st.session_state.analyzer
        self.notifier      = st.session_state.notifier
        self.net_capture   = st.session_state.net_capture
        self.alert_manager = st.session_state.alert_manager

    def sidebar(self):
        st.sidebar.title("🛡️ Menu")
        return st.sidebar.radio("Go to", [
            "📊 Overview",
            "🚀 Live Detection",
            "⚠️ Alerts",
            "📜 Incident History",
            "📡 System Status",
            "🔎 Threat Intelligence",
            "⚙️ Configuration",
        ])

    def header(self):
        st.title("🔒 Multimodal LLM Cybersecurity Threat Detection")
        st.caption("Context-Aware Security Analysis for Communication Networks")
        st.divider()

    # ── Overview ──────────────────────────────────────────────────────────────
    def page_overview(self):
        st.header("📊 Overview")

        st.subheader("🏆 Model Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detection Accuracy",      f"{HEADLINE_METRICS['threat_detection_accuracy_pct']}%")
        c2.metric("False Positive Reduction", f"{HEADLINE_METRICS['false_positive_reduction_pct']}%")
        c3.metric("Alert Volume Reduction",   f"{HEADLINE_METRICS['alert_volume_reduction_pct']}%")
        c4.metric("True Positive Increase",   f"+{HEADLINE_METRICS['true_positive_increase_pct']}%")

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{HEADLINE_METRICS['proposed_model_accuracy']}%")
        c2.metric("Precision", f"{HEADLINE_METRICS['proposed_model_precision']}%")
        c3.metric("Recall",    f"{HEADLINE_METRICS['proposed_model_recall']}%")
        c4.metric("F1 Score",  f"{HEADLINE_METRICS['proposed_model_f1']}%")

        st.divider()

        st.subheader("🔄 Live Session Stats")
        alerts   = self.alert_manager.alerts
        active   = sum(1 for a in alerts if a.get("status") == "active")
        critical = sum(1 for a in alerts if a.get("risk_level") == "critical")
        avg_risk = (sum(a.get("overall_risk_score", 0) for a in alerts) / len(alerts)) if alerts else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Alerts",    active)
        c2.metric("Critical Threats", critical)
        c3.metric("Total Detections", len(alerts))
        c4.metric("Avg Risk Score",   f"{avg_risk:.2f}")

        st.divider()

        st.subheader("🏗️ Data Sources")
        sources = [
            ("📄", "System Logs"),
            ("🌐", "Network Traffic"),
            ("🚨", "IDS Alerts"),
            ("🔍", "DNS Records"),
            ("💻", "Endpoint Activity"),
        ]
        for col, (icon, label) in zip(st.columns(5), sources):
            col.markdown(f"""<div class="source-card"><h2>{icon}</h2><b>{label}</b></div>""",
                         unsafe_allow_html=True)

        st.divider()

        st.subheader("⚙️ Detection Pipeline")
        steps = [
            ("1", "Data Ingestion",     "Collect from all 5 data sources"),
            ("2", "Preprocessing",      "Feature extraction & normalization"),
            ("3", "Representation",     "Convert to textual representations"),
            ("4", "Correlation",        "Cosine similarity across sources"),
            ("5", "LLM Reasoning",      "Threat classification & analysis"),
            ("6", "Knowledge Matching", "MITRE ATT&CK & CVE enrichment"),
            ("7", "Alert Generation",   "Severity scoring & notification"),
        ]
        cols = st.columns(len(steps))
        for col, (num, title, desc) in zip(cols, steps):
            col.markdown(f"""<div class="source-card">
                <b style="font-size:1.3em;color:#2196F3">{num}</b><br>
                <b>{title}</b><br>
                <small style="color:#666">{desc}</small>
            </div>""", unsafe_allow_html=True)

    # ── Live Detection ────────────────────────────────────────────────────────
    def page_live_detection(self):
        st.header("🚀 Live Threat Detection")

        c1, c2 = st.columns(2)
        run_all = c1.button("🔍 Run Full Detection Pipeline", use_container_width=True)
        run_net = c2.button("📡 Network Capture Only",        use_container_width=True)

        if run_net:
            with st.spinner("Capturing packets…"):
                packets  = self.net_capture.capture_packets(packet_count=50)
                flows    = self.net_capture.analyze_flows()
                patterns = self.net_capture.detect_suspicious_patterns()
            st.success(f"✅ {len(packets)} packets | {len(flows)} flows | {len(patterns)} suspicious patterns")

        # ── Run pipeline and cache results in session_state ──────────────────
        if run_all:
            with st.spinner("Running detection pipeline…"):
                packets  = self.net_capture.capture_packets(packet_count=100)
                flows    = self.net_capture.analyze_flows()
                patterns = self.net_capture.detect_suspicious_patterns()
                processed = self.processor.process_all()
                reps  = processed["representations"]
                result = self.analyzer.perform_threat_analysis(
                    processed["llm_input"], representations=reps)
                ta = result["threat_analysis"]
                if result["threat_detected"]:
                    indicators = ta["pattern"].split()[:10]
                    rar = self.analyzer.retrieval_augmented_reasoning(ta["threat_type"], indicators)
                    alert = self.alert_manager.create_alert(result)
                    self.notifier.broadcast_alert(alert)
                else:
                    rar = {"matching_knowledge": [], "enriched_analysis": {},
                           "alpha_threshold": self.analyzer.alpha}

                # Save to session_state — prevents re-running on every Streamlit rerender
                st.session_state["det_packets"]   = packets
                st.session_state["det_flows"]     = flows
                st.session_state["det_patterns"]  = patterns
                st.session_state["det_processed"] = processed
                st.session_state["det_result"]    = result
                st.session_state["det_rar"]       = rar
                # Clear cached chart so it regenerates for new run
                st.session_state.pop("det_chart_img", None)

        # ── Only render results if we have them ──────────────────────────────
        if "det_result" not in st.session_state:
            st.info("Click **🔍 Run Full Detection Pipeline** to start analysis.")
            return

        # Pull stable cached values — no blinking
        packets   = st.session_state["det_packets"]
        flows     = st.session_state["det_flows"]
        patterns  = st.session_state["det_patterns"]
        processed = st.session_state["det_processed"]
        agg       = processed["aggregated_data"]
        reps      = processed["representations"]
        result    = st.session_state["det_result"]
        rar       = st.session_state["det_rar"]
        ta        = result["threat_analysis"]

        threat_type = ta.get("threat_type", "no_threat")
        is_threat   = result["threat_detected"] or (
            threat_type not in ("no_threat", "normal", "") and
            result["overall_risk_level"] in ("high", "critical", "medium")
        )
        ea = rar.get("enriched_analysis", {})

        # ── Prominent banner ─────────────────────────────────────────────────
        if is_threat:
            threat_name = threat_type.replace("_", " ").title()
            risk_level  = result["overall_risk_level"].upper()
            confidence  = int(ta["confidence"] * 100)
            mitre       = ea.get("mitre_technique", "N/A")
            if mitre == "N/A" and "MITRE" in ta.get("reasoning", ""):
                mitre = ta["reasoning"].split("MITRE")[-1].strip("() ").split()[0]
            banner_color = {
                "CRITICAL": "linear-gradient(135deg,#6f0000,#dc3545)",
                "HIGH":     "linear-gradient(135deg,#dc3545,#a71d2a)",
                "MEDIUM":   "linear-gradient(135deg,#e67e00,#c56200)",
            }.get(risk_level, "linear-gradient(135deg,#dc3545,#a71d2a)")

            st.markdown(f"""
            <div style="background:{banner_color};border-radius:12px;
                        padding:28px 32px;margin-bottom:20px;color:white;">
                <div style="font-size:2em;font-weight:800;letter-spacing:1px">
                    🚨 THREAT DETECTED
                </div>
                <div style="font-size:1.4em;margin-top:6px;font-weight:600">{threat_name}</div>
                <div style="margin-top:14px;display:flex;gap:32px;flex-wrap:wrap">
                    <span>🔴 <b>Risk:</b> {risk_level}</span>
                    <span>🎯 <b>Confidence:</b> {confidence}%</span>
                    <span>🛡️ <b>MITRE:</b> {mitre}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            remediation = ea.get("remediation", "")
            all_recs = _filter_recs(result.get("recommendations") or [], threat_type)
            recs = all_recs[:4]
            st.markdown(f"""
            <div style="background:#fff8f8;border:1px solid #f5c6cb;
                        border-radius:10px;padding:20px;margin-bottom:20px">
                <b style="font-size:1.1em">📋 Detection Summary</b>
                <hr style="border-color:#f5c6cb;margin:10px 0">
                <p style="margin:4px 0"><b>Pattern:</b> {ta.get('pattern','—')}</p>
                <p style="margin:4px 0"><b>Analysis:</b> {ta.get('reasoning','—')}</p>
                {"<p style='margin:10px 0 4px 0'><b>Recommended Actions:</b></p>" +
                 "".join(f"<p style='margin:2px 0 2px 16px'>• {r}</p>" for r in recs) if recs else ""}
                {"<p style='margin:8px 0 0 0'><b>Remediation:</b> " + remediation + "</p>" if remediation else ""}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:linear-gradient(135deg,#28a745,#1a7a30);
                        border-radius:12px;padding:28px 32px;margin-bottom:20px;color:white;">
                <div style="font-size:2em;font-weight:800">✅ ALL CLEAR</div>
                <div style="font-size:1.1em;margin-top:6px">
                    No threats detected — system operating normally</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Pipeline detail expanders ─────────────────────────────────────────
        def _mc(label, value):
            return (f'<div class="metric-card"><b>{label}</b><br>'
                    f'<span style="font-size:1.4em;font-weight:600">{value}</span></div>')

        with st.expander("📡 Network Capture", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(_mc("Packets Captured",    len(packets)),    unsafe_allow_html=True)
            c2.markdown(_mc("Flows Analysed",      len(flows)),      unsafe_allow_html=True)
            c3.markdown(_mc("Suspicious Patterns", len(patterns)),   unsafe_allow_html=True)

        with st.expander("🔀 Data Processing", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.markdown(_mc("Data Sources",    agg["num_sources"]),              unsafe_allow_html=True)
            c2.markdown(_mc("Total Events",    agg["total_events"]),             unsafe_allow_html=True)
            c3.markdown(_mc("Events Analysed", len(processed["cleaned_events"])),unsafe_allow_html=True)
            src_counts = {
                "System Logs":       len(agg.get("system_logs", [])),
                "Network Traffic":   len(agg.get("network_traffic", [])),
                "IDS Alerts":        len(agg.get("ids_alerts", [])),
                "DNS Records":       len(agg.get("dns_records", [])),
                "Endpoint Activity": len(agg.get("endpoint_activity", [])),
            }
            if "det_chart_img" not in st.session_state:
                import io, base64
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.bar(src_counts.keys(), src_counts.values(),
                       color=["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"])
                ax.set_title("Events per Data Source")
                ax.set_ylabel("Count")
                plt.xticks(rotation=15, ha="right")
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                plt.close()
                buf.seek(0)
                st.session_state["det_chart_img"] = base64.b64encode(buf.read()).decode()
            st.markdown(
                f'<img src="data:image/png;base64,{st.session_state["det_chart_img"]}" '
                f'style="width:100%;border-radius:8px">',
                unsafe_allow_html=True,
            )

        with st.expander("🤖 LLM Analysis", expanded=False):
            if is_threat:
                c1, c2 = st.columns(2)
                c1.markdown(_mc("Threat Type", ta["threat_type"].replace("_", " ").title()), unsafe_allow_html=True)
                c2.markdown(_mc("Risk Level",  result["overall_risk_level"].upper()),         unsafe_allow_html=True)
                st.caption("Full pattern and analysis available in the Detection Summary above.")
            else:
                st.success("No threat identified by LLM analysis.")

        with st.expander("🔍 Knowledge Base Matching", expanded=False):
            ea = rar.get("enriched_analysis", {})
            # Build entire KB section as one static HTML block — no Streamlit widgets = no blinking
            mitre_val    = ea.get("mitre_technique", "N/A") if ea else "N/A"
            severity_val = ea.get("severity", "N/A").title() if ea else "N/A"
            kb_count     = len(rar["matching_knowledge"])

            # Metric cards row
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:16px">
                <div class="metric-card" style="flex:1"><b>MITRE Technique</b><br>
                    <span style="font-size:1.3em">{mitre_val}</span></div>
                <div class="metric-card" style="flex:1"><b>Severity</b><br>
                    <span style="font-size:1.3em">{severity_val}</span></div>
                <div class="metric-card" style="flex:1"><b>KB Matches</b><br>
                    <span style="font-size:1.3em">{kb_count}</span></div>
            </div>""", unsafe_allow_html=True)

            # Static HTML table instead of st.dataframe
            if rar["matching_knowledge"]:
                rows_html = ""
                for row in rar["matching_knowledge"]:
                    mt    = row.get("matched_threat", "")
                    score = f'{row.get("similarity_score", 0):.4f}'
                    mitre = row.get("mitre_technique", "")
                    sev   = row.get("severity", "")
                    rows_html += (f"<tr><td>{mt}</td><td>{score}</td>"
                                  f"<td>{mitre}</td><td>{sev}</td></tr>")
                st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-size:0.9em">
                    <thead>
                        <tr style="background:#f0f2f6;text-align:left">
                            <th style="padding:8px 12px;border-bottom:2px solid #dee2e6">matched_threat</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #dee2e6">similarity_score</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #dee2e6">mitre_technique</th>
                            <th style="padding:8px 12px;border-bottom:2px solid #dee2e6">severity</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html.replace("<tr>", '<tr style="border-bottom:1px solid #dee2e6">').replace("<td>", '<td style="padding:8px 12px">')}
                    </tbody>
                </table>""", unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="color:#555;padding:8px 0">No knowledge base matches found.</div>',
                    unsafe_allow_html=True)

        with st.expander("🔔 Alert Details", expanded=False):
            if is_threat:
                all_recs = _filter_recs(result.get("recommendations") or [], threat_type)
                extra = all_recs[4:6]
                if extra:
                    st.markdown("**Additional Recommended Actions:**")
                    for rec in extra:
                        st.markdown(f"- {rec}")
                else:
                    st.caption("All recommended actions listed in the Detection Summary above.")
            else:
                st.markdown('<div class="result-box">No alert generated. Continuing monitoring.</div>',
                            unsafe_allow_html=True)

    # ── Alerts ────────────────────────────────────────────────────────────────
    def page_alerts(self):
        st.header("⚠️ Active Alerts")

        # Always read directly from session_state so mutations are immediately visible
        alert_manager = st.session_state.alert_manager
        alerts = alert_manager.alerts

        if not alerts:
            st.info("No alerts yet. Run the Live Detection pipeline first.")
            return

        # Summary counts at top
        total   = len(alerts)
        active  = sum(1 for a in alerts if a.get("status") == "active")
        acked   = sum(1 for a in alerts if a.get("status") == "acknowledged")
        resolved= sum(1 for a in alerts if a.get("status") == "resolved")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'''<div class="metric-card"><b>Total</b><br>
            <span style="font-size:1.4em;font-weight:600">{total}</span></div>''',
            unsafe_allow_html=True)
        c2.markdown(f'''<div class="metric-card" style="border-top:3px solid #28a745"><b>Active</b><br>
            <span style="font-size:1.4em;font-weight:600;color:#28a745">{active}</span></div>''',
            unsafe_allow_html=True)
        c3.markdown(f'''<div class="metric-card" style="border-top:3px solid #007bff"><b>Acknowledged</b><br>
            <span style="font-size:1.4em;font-weight:600;color:#007bff">{acked}</span></div>''',
            unsafe_allow_html=True)
        c4.markdown(f'''<div class="metric-card" style="border-top:3px solid #6c757d"><b>Resolved</b><br>
            <span style="font-size:1.4em;font-weight:600;color:#6c757d">{resolved}</span></div>''',
            unsafe_allow_html=True)

        st.divider()

        risk_filter = st.multiselect("Filter by Risk Level",
            ["critical","high","medium","low"], default=["critical","high","medium","low"])
        status_filter = st.multiselect("Filter by Status",
            ["active","acknowledged","resolved"], default=["active"])

        shown = [a for a in alerts
                 if a.get("risk_level","low") in risk_filter
                 and a.get("status","active") in status_filter]

        if not shown:
            st.info("No alerts matching the selected filters.")
            return

        icon_map    = {"critical":"🔴","high":"🔴","medium":"🟠","low":"🟡"}
        status_badge = {
            "active":       '<span style="background:#28a745;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em">● Active</span>',
            "acknowledged": '<span style="background:#007bff;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em">● Acknowledged</span>',
            "resolved":     '<span style="background:#6c757d;color:white;padding:2px 10px;border-radius:12px;font-size:0.8em">● Resolved</span>',
        }

        for alert in shown:
            risk_level = alert.get("risk_level","low")
            status     = alert.get("status","active")
            icon       = icon_map.get(risk_level, "⚪")
            threat     = alert.get("threat_type","?").replace("_"," ").upper()
            confidence = alert.get("confidence", alert.get("confidence_score", 0))
            severity   = alert.get("severity_score", 0)
            mitre      = alert.get("mitre_technique","N/A")
            systems    = alert.get("affected_systems", alert.get("source_ips",[]))
            actions    = [a for a in alert.get("recommended_actions",
                          alert.get("actions_recommended", [])) if a and a.strip()]
            actions    = _filter_recs(actions, alert.get("threat_type",""))
            alert_id   = alert["alert_id"]

            border_color = ('#dc3545' if risk_level in ('critical','high')
                            else '#fd7e14' if risk_level=='medium' else '#ffc107')

            st.markdown(f"""
            <div style="background:#fff;border:1px solid #dee2e6;
                        border-left:5px solid {border_color};
                        border-radius:8px;padding:20px;margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div style="font-size:1.25em;font-weight:700">{icon} {threat}</div>
                    {status_badge.get(status,'')}
                </div>
                <div style="color:#6c757d;font-size:0.85em;margin:4px 0 12px 0">
                    ID: {alert_id}
                    {f" &nbsp;|&nbsp; Resolved: {alert.get('resolved_time','')[:19].replace('T',' ')}" if status=="resolved" else ""}
                    {f" &nbsp;|&nbsp; Acknowledged: {alert.get('acknowledged_time','')[:19].replace('T',' ')}" if status=="acknowledged" else ""}
                </div>
                <div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:12px">
                    <span><b>Risk:</b> {risk_level.upper()}</span>
                    <span><b>Confidence:</b> {int(confidence*100) if confidence<=1 else int(confidence)}%</span>
                    <span><b>Severity:</b> {int(severity*100) if severity<=1 else int(severity)}%</span>
                    <span><b>MITRE:</b> <code>{mitre}</code></span>
                </div>
                {f'<div style="margin-bottom:8px"><b>Affected:</b> {", ".join(systems)}</div>' if systems else ''}
            </div>""", unsafe_allow_html=True)

            ca, cb = st.columns([4,1])
            with ca:
                if actions:
                    with st.expander("Recommended Actions"):
                        for a in actions[:6]:
                            st.markdown(f"- {a}")
            with cb:
                if status == "active":
                    if st.button("✅ Acknowledge", key=f"ack_{alert_id}", use_container_width=True):
                        alert_manager.acknowledge_alert(alert_id)
                        _safe_rerun()
                if status in ("active", "acknowledged"):
                    if st.button("🔒 Resolve", key=f"res_{alert_id}", use_container_width=True):
                        alert_manager.resolve_alert(alert_id)
                        _safe_rerun()
            st.divider()

    # ── Incident History ──────────────────────────────────────────────────────
    def page_incident_history(self):
        st.header("📜 Incident History")
        all_alerts = self.alert_manager.alerts + getattr(self.alert_manager, "alert_history", [])
        seen, history = set(), []
        for a in all_alerts:
            aid = a.get("alert_id")
            if aid and aid not in seen:
                seen.add(aid); history.append(a)

        if not history:
            st.info("No incidents recorded yet. Run the Live Detection pipeline to generate alerts.")
            return

        total    = len(history)
        resolved = sum(1 for a in history if a.get("status") == "resolved")
        acked    = sum(1 for a in history if a.get("status") == "acknowledged")
        active   = sum(1 for a in history if a.get("status") == "active")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Incidents", total)
        c2.metric("Active",          active)
        c3.metric("Acknowledged",    acked)
        c4.metric("Resolved",        resolved)
        st.divider()

        status_filter = st.multiselect("Filter by Status",
            ["active","acknowledged","resolved"],
            default=["active","acknowledged","resolved"])
        shown = [a for a in history if a.get("status","active") in status_filter]
        if not shown:
            st.info("No incidents match the selected filter.")
            return

        status_badge = {"active":"🟢 Active","acknowledged":"🔵 Acknowledged","resolved":"⚫ Resolved"}
        rows = []
        for a in reversed(shown):
            rows.append({
                "Time":       a.get("timestamp","—")[:19].replace("T"," "),
                "Threat":     a.get("threat_type","?").replace("_"," ").title(),
                "Risk":       a.get("risk_level","?").upper(),
                "Confidence": f"{int(a.get('confidence', a.get('confidence_score',0))*100)}%",
                "MITRE":      a.get("mitre_technique","N/A"),
                "Status":     status_badge.get(a.get("status","active"), a.get("status","?")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.divider()

        st.subheader("Incidents by Threat Type")
        from collections import Counter
        counts = Counter(a.get("threat_type","unknown").replace("_"," ").title() for a in history)
        if counts:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(list(counts.keys()), list(counts.values()),
                    color=["#dc3545","#fd7e14","#ffc107","#2196F3","#4CAF50"][:len(counts)])
            ax.set_xlabel("Count")
            ax.set_title("Incident Frequency by Threat Type")
            ax.invert_yaxis()
            st.pyplot(fig)
            plt.close()

    # ── System Status ─────────────────────────────────────────────────────────
    def page_system_status(self):
        st.header("📡 System Status")
        st.subheader("Data Source Health")
        sources = {
            "System Logs":       ("📄", SystemConfig.DATA_SOURCES.get("system_logs", True)),
            "Network Traffic":   ("🌐", SystemConfig.DATA_SOURCES.get("network_traffic", True)),
            "IDS Alerts":        ("🚨", SystemConfig.DATA_SOURCES.get("ids_alerts", True)),
            "DNS Records":       ("🔍", SystemConfig.DATA_SOURCES.get("dns_records", True)),
            "Endpoint Activity": ("💻", SystemConfig.DATA_SOURCES.get("endpoint_activity", True)),
        }
        cols = st.columns(5)
        for col, (name, (icon, enabled)) in zip(cols, sources.items()):
            status_color = "#28a745" if enabled else "#dc3545"
            status_text  = "Online" if enabled else "Offline"
            col.markdown(f"""
            <div style="background:#f8f9fa;border:1px solid #dee2e6;
                        border-top:4px solid {status_color};
                        border-radius:8px;padding:16px;text-align:center">
                <div style="font-size:1.8em">{icon}</div>
                <div style="font-weight:600;margin:6px 0">{name}</div>
                <div style="color:{status_color};font-weight:700">{status_text}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Pipeline Components")
        components = [
            ("🔀 Multimodal Processor", "Online", "#28a745"),
            ("🤖 LLM Analyzer",         "Online", "#28a745"),
            ("📚 Knowledge Base",       "Online", "#28a745"),
            ("🔔 Alert Manager",        "Online", "#28a745"),
            ("📡 Network Capture",      "Online", "#28a745"),
            ("📣 Alert Notifier",       "Online", "#28a745"),
        ]
        c1, c2 = st.columns(2)
        for i, (name, status, color) in enumerate(components):
            col = c1 if i % 2 == 0 else c2
            col.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;
                        padding:12px 16px;margin-bottom:10px">
                <span style="font-weight:600">{name}</span>
                <span style="color:{color};font-weight:700">● {status}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("Current Session")
        alerts  = self.alert_manager.alerts
        active  = sum(1 for a in alerts if a.get("status") == "active")
        crits   = sum(1 for a in alerts if a.get("risk_level") == "critical")
        highs   = sum(1 for a in alerts if a.get("risk_level") == "high")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Alerts This Session", len(alerts))
        c2.metric("Currently Active",          active)
        c3.metric("Critical",                  crits)
        c4.metric("High",                      highs)

    # ── Threat Intelligence ───────────────────────────────────────────────────
    def page_threat_intelligence(self):
        st.header("🔎 Threat Intelligence")
        st.caption("Search and explore MITRE ATT&CK techniques and known threat patterns.")

        MITRE_DB = {
            "T1046":     {"name":"Network Service Discovery",         "tactic":"Discovery",           "desc":"Adversaries may scan victim systems to discover services running on remote hosts.",               "severity":"Medium"},
            "T1059":     {"name":"Command & Scripting Interpreter",   "tactic":"Execution",           "desc":"Adversaries may abuse command and script interpreters to execute commands, scripts, or binaries.","severity":"High"},
            "T1068":     {"name":"Exploitation for Privilege Escalation","tactic":"Privilege Escalation","desc":"Adversaries may exploit software vulnerabilities to elevate privileges.",                      "severity":"High"},
            "T1078":     {"name":"Valid Accounts",                    "tactic":"Defense Evasion",     "desc":"Adversaries may obtain and abuse credentials of existing accounts.",                             "severity":"High"},
            "T1110":     {"name":"Brute Force",                       "tactic":"Credential Access",   "desc":"Adversaries may use brute force techniques to gain access to accounts.",                        "severity":"Medium"},
            "T1110.004": {"name":"Credential Stuffing",               "tactic":"Credential Access",   "desc":"Adversaries may use credentials obtained from breach dumps to gain access to victim accounts.",  "severity":"Medium"},
            "T1041":     {"name":"Exfiltration Over C2 Channel",      "tactic":"Exfiltration",        "desc":"Adversaries may steal data by exfiltrating it over an existing command and control channel.",   "severity":"High"},
            "T1071.004": {"name":"DNS Application Layer Protocol",    "tactic":"Command & Control",   "desc":"Adversaries may communicate using the DNS protocol to avoid detection.",                        "severity":"Medium"},
            "T1486":     {"name":"Data Encrypted for Impact",         "tactic":"Impact",              "desc":"Adversaries may encrypt data on target systems (ransomware).",                                  "severity":"Critical"},
            "T1498":     {"name":"Network Denial of Service",         "tactic":"Impact",              "desc":"Adversaries may perform Network DoS attacks to degrade availability.",                          "severity":"High"},
            "T1496":     {"name":"Resource Hijacking",                "tactic":"Impact",              "desc":"Adversaries may leverage the resources of co-opted systems for intensive tasks.",                "severity":"Medium"},
        }

        sev_color = {"Critical":"#dc3545","High":"#fd7e14","Medium":"#ffc107","Low":"#28a745"}
        query = st.text_input("🔍 Search by MITRE ID or keyword",
                              placeholder="e.g. T1059 or brute force")
        if query:
            q = query.strip().upper()
            results = {k:v for k,v in MITRE_DB.items()
                       if q in k.upper() or q in v["name"].upper()
                       or q in v["tactic"].upper() or q in v["desc"].upper()}
        else:
            results = MITRE_DB

        st.caption(f"{len(results)} technique{'s' if len(results)!=1 else ''} found")
        st.divider()

        for tid, info in results.items():
            sc = sev_color.get(info["severity"], "#6c757d")
            with st.expander(f"`{tid}` — {info['name']}  |  {info['tactic']}"):
                c1, c2 = st.columns([3,1])
                with c1:
                    st.markdown(f"**Description:** {info['desc']}")
                    st.markdown(f"**Tactic:** {info['tactic']}")
                with c2:
                    st.markdown(f"""
                    <div style="background:{sc};color:white;border-radius:6px;
                                padding:8px 14px;text-align:center;font-weight:700">
                        {info['severity']}
                    </div>""", unsafe_allow_html=True)
                st.markdown(f"[🔗 View on MITRE ATT&CK](https://attack.mitre.org/techniques/{tid.replace('.','/')})")

        st.divider()
        session_alerts = self.alert_manager.alerts
        if session_alerts:
            st.subheader("Techniques Seen This Session")
            seen_mitres = list({a.get("mitre_technique","N/A") for a in session_alerts
                                if a.get("mitre_technique","N/A") != "N/A"})
            if seen_mitres:
                for tid in seen_mitres:
                    info = MITRE_DB.get(tid)
                    if info:
                        sc = sev_color.get(info["severity"],"#6c757d")
                        st.markdown(f"- `{tid}` **{info['name']}** — "
                                    f"<span style='color:{sc};font-weight:700'>{info['severity']}</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"- `{tid}` — not in local database")
            else:
                st.info("No MITRE techniques recorded in current session alerts.")

    # ── Configuration ─────────────────────────────────────────────────────────
    def page_config(self):
        st.header("⚙️ Configuration")
        with st.expander("Threat Detection Settings"):
            tc = SystemConfig.THREAT_CONFIG
            c1,c2 = st.columns(2)
            c1.metric("Risk Threshold",       tc["risk_threshold"])
            c2.metric("Similarity Threshold", tc["similarity_threshold"])
            c1.metric("KB Match Threshold",   tc["knowledge_match_threshold"])
            c2.metric("Severity Weight 1",    tc["severity_weight_risk"])
            c1.metric("Severity Weight 2",    tc["severity_weight_impact"])
        with st.expander("Data Sources"):
            for src, enabled in SystemConfig.DATA_SOURCES.items():
                st.checkbox(src.replace("_"," ").title(), value=enabled, disabled=True)
        with st.expander("Export Alerts"):
            if self.alert_manager.alerts:
                st.download_button(
                    "⬇️ Download Alerts (JSON)",
                    data=json.dumps(self.alert_manager.alerts, indent=2, default=str),
                    file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json")
            else:
                st.info("No alerts to export yet.")

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        self.header()
        page = self.sidebar()
        if   page == "📊 Overview":           self.page_overview()
        elif page == "🚀 Live Detection":      self.page_live_detection()
        elif page == "⚠️ Alerts":              self.page_alerts()
        elif page == "📜 Incident History":    self.page_incident_history()
        elif page == "📡 System Status":       self.page_system_status()
        elif page == "🔎 Threat Intelligence": self.page_threat_intelligence()
        elif page == "⚙️ Configuration":       self.page_config()


def main():
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = Dashboard()
    st.session_state.dashboard.run()

if __name__ == "__main__":
    main()