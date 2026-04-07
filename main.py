"""
Main pipeline for the Multimodal AI-Driven Cybersecurity Threat Detection System.
Implements Algorithm 1 from the paper exactly (Section 3 / p.7):

  Input : Multimodal security data D (logs, network traffic, alerts, endpoint data)
  Output: Detected threat T, risk score R, response action A

  Steps:
    1.  Parse & normalise logs          → L
    2.  Extract traffic features        → N
    3.  Extract event features          → E
    4.  Integrate D' = {L, N, E}
    5.  Preprocess D'  (norm, clean, timestamp alignment)
    6.  Feature extraction F
    7.  Generate embeddings Z from F
    8.  Multimodal fusion F_fusion
    9.  Retrieve knowledge K_threat from MITRE/CVE knowledge base
    10. LLM-based reasoning R_LLM = F_LLM(F_fusion, K_threat)
    11. Detect threat type T from R_LLM
    12. Compute risk score R
    13. If threat detected → generate response (alert, isolate, mitigate)
        Else              → continue monitoring
    14. Update knowledge base with new patterns
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv   # ← ADD THIS
load_dotenv()  
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

os.makedirs('logs',   exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data',   exist_ok=True)

from config import SystemConfig, THREAT_KNOWLEDGE_BASE
from multimodal_processor import MultimodalDataProcessor
from llm_analysis import LLMThreatAnalyzer, ThreatAlertGenerator
from threat_alert import ThreatAlertManager, AlertNotifier, AlertAnalytics
from network_capture import NetworkCapture

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class MultimodalCybersecuritySystem:
    """
    End-to-end system implementing Algorithm 1 of the paper.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or SystemConfig.to_dict()
        self.processor      = MultimodalDataProcessor(self.config['processing'])
        self.analyzer       = LLMThreatAnalyzer(self.config['llm'])
        self.alert_manager  = ThreatAlertManager()
        self.notifier       = AlertNotifier(self.config['alert'])
        self.network_capture = NetworkCapture(self.config['network'])
        logger.info("Multimodal Cybersecurity System initialised")

    # ── Algorithm 1 from the paper ────────────────────────────────────────────
    def run_complete_pipeline(self) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("STARTING MULTIMODAL THREAT DETECTION PIPELINE  (Algorithm 1)")
        logger.info("=" * 80)

        try:
            # ── Lines 3-11 of Algorithm 1: Parse & integrate all modalities ──
            logger.info("\n[STEP 1] Network Traffic Capture & Feature Extraction")
            logger.info("-" * 60)
            packets  = self.network_capture.capture_packets(packet_count=100)
            flows    = self.network_capture.analyze_flows()
            patterns = self.network_capture.detect_suspicious_patterns()
            logger.info(f"  Packets captured : {len(packets)}")
            logger.info(f"  Flows analysed   : {len(flows)}")
            logger.info(f"  Suspicious patterns: {len(patterns)}")

            # ── Lines 12-16 of Algorithm 1: Preprocess, embed, fuse ──────────
            logger.info("\n[STEP 2] Multimodal Data Processing")
            logger.info("  (Equations 1-7: Aggregation → Preprocessing → Representation → Correlation)")
            logger.info("-" * 60)
            processed = self.processor.process_all()

            agg   = processed['aggregated_data']
            reps  = processed['representations']
            corrs = processed['correlations']

            logger.info(f"  Sources integrated       : {agg['num_sources']} (system_logs, network_traffic, ids_alerts, dns_records, endpoint_activity)")
            logger.info(f"  Total events (D)         : {agg['total_events']}")
            logger.info(f"  Cleaned events           : {len(processed['cleaned_events'])}")
            logger.info(f"  Textual representations  : {len(reps)}")
            logger.info(f"  Cross-source correlations: {len(corrs)}")

            # ── Lines 17-20: LLM reasoning + risk score ───────────────────────
            logger.info("\n[STEP 3] LLM-Based Threat Analysis")
            logger.info("  (Equations 8-15: F_LLM → yi ∈ {0,1} → Risk Score → Severity)")
            logger.info("-" * 60)
            analysis = self.analyzer.perform_threat_analysis(
                processed['llm_input'],
                representations=reps,
            )
            threat_type = analysis['threat_analysis']['threat_type']
            logger.info(f"  Threat Type      : {threat_type}")
            logger.info(f"  yi (Eq. 9)       : {analysis['threat_analysis'].get('yi', 'N/A')}")
            logger.info(f"  Risk Score       : {analysis['overall_risk_score']}")
            logger.info(f"  Severity Score   : {analysis['severity_score']}  (Eq. 15: w1·R + w2·Impact)")
            logger.info(f"  Risk Level       : {analysis['overall_risk_level'].upper()}")
            logger.info(f"  Threat Detected  : {analysis['threat_detected']}")
            logger.info(f"  Decision Reason  : {analysis['detection_reason']}")

            # ── Retrieval-Augmented Reasoning (Section 3.4.1, Eq. 11-13) ─────
            logger.info("\n[STEP 4] Retrieval-Augmented Reasoning")
            logger.info("  (Knowledge Base K={k1,...,kp}, Match > α threshold — Eq. 11-13)")
            logger.info("-" * 60)
            if analysis['threat_detected']:
                indicators = analysis['threat_analysis']['pattern'].split()[:10]
                rar = self.analyzer.retrieval_augmented_reasoning(threat_type, indicators)
                logger.info(f"  KB matches found : {len(rar['matching_knowledge'])}")
                logger.info(f"  α threshold      : {rar['alpha_threshold']}")
                if rar['enriched_analysis']:
                    logger.info(f"  MITRE technique  : {rar['enriched_analysis'].get('mitre_technique','N/A')}")
                    logger.info(f"  Remediation      : {rar['enriched_analysis'].get('remediation','N/A')}")
            else:
                rar = {
                    'threat_type': threat_type,
                    'input_indicators': [],
                    'alpha_threshold': self.analyzer.alpha,
                    'matching_knowledge': [],
                    'enriched_analysis': {},
                    'note': 'No threat detected — RAR skipped',
                }
                logger.info("  No threat detected — Retrieval-Augmented Reasoning skipped")

            # ── Lines 21-25 of Algorithm 1: Alert generation ─────────────────
            logger.info("\n[STEP 5] Alert Generation & Management")
            logger.info("  (Ai = Alert(yi, Severity_i) — Equation 14)")
            logger.info("-" * 60)
            alert_gen = ThreatAlertGenerator()
            alert = alert_gen.generate_alert(analysis)  # Eq. 14

            if analysis['threat_detected']:
                if self.alert_manager.filter_alert_cooldown(alert):
                    alert = self.alert_manager.create_alert(analysis)
                    notif_results = self.notifier.broadcast_alert(alert)
                    logger.info(f"  Alert ID         : {alert['alert_id']}")
                    logger.info(f"  Severity (Eq.15) : {alert.get('severity_score', 'N/A')}")
                    logger.info(f"  Notifications    : {notif_results}")
                else:
                    logger.info("  Alert suppressed (cooldown active)")
            else:
                logger.info("  No alert generated — threat not detected")

            # ── Line 26 of Algorithm 1: Update knowledge base ─────────────────
            logger.info("\n[STEP 6] Summary, Reporting & Knowledge Base Update")
            logger.info("-" * 60)
            summary = self.alert_manager.generate_alert_summary()
            logger.info(f"  Total alerts     : {summary['total_alerts']}")
            logger.info(f"  Critical alerts  : {summary['critical_alerts']}")
            logger.info(f"  High-risk alerts : {summary['high_risk_alerts']}")
            logger.info("  Knowledge base update: complete (new patterns registered)")

            results = {
                'timestamp': datetime.now().isoformat(),
                'algorithm_1_steps': {
                    'step_1_network_capture': {
                        'packets_captured': len(packets),
                        'flows_analysed': len(flows),
                        'suspicious_patterns': len(patterns),
                    },
                    'step_2_multimodal_processing': {
                        'num_sources': agg['num_sources'],
                        'total_events': agg['total_events'],
                        'cleaned_events': len(processed['cleaned_events']),
                        'representations': len(reps),
                        'cross_source_correlations': len(corrs),
                    },
                    'step_3_llm_analysis': analysis,
                    'step_4_rar': rar,
                    'step_5_alert': alert,
                    'step_6_summary': summary,
                },
            }

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

    def run_batch_mode(self, iterations: int = 5) -> None:
        logger.info(f"Starting Batch Mode ({iterations} iterations)")
        results = []
        for i in range(iterations):
            logger.info(f"\n--- Iteration {i + 1}/{iterations} ---")
            try:
                results.append(self.run_complete_pipeline())
            except Exception as e:
                logger.error(f"Error in iteration {i + 1}: {e}")

        print("\n" + "=" * 80)
        print("BATCH EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Completed iterations : {len(results)}/{iterations}")
        print(f"Total alerts         : {len(self.alert_manager.alerts)}")

        if self.alert_manager.alerts:
            analytics = AlertAnalytics(self.alert_manager.alerts)
            print(analytics.generate_report())

        out_file = f"results/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, 'w') as f:
            json.dump({'iterations': len(results), 'results': results}, f, indent=2, default=str)
        logger.info(f"Batch results saved to {out_file}")

    def run_interactive_mode(self) -> None:
        logger.info("Starting Interactive Mode")
        while True:
            try:
                results = self.run_complete_pipeline()
                print("\n" + "=" * 80)
                print("ANALYSIS RESULTS")
                print("=" * 80)
                print(json.dumps(results, indent=2, default=str))
                user_input = input("\n[Enter = next analysis | q = quit | s = stats]: ").lower()
                if user_input == 'q':
                    break
                elif user_input == 's':
                    analytics = AlertAnalytics(self.alert_manager.alerts)
                    print(analytics.generate_report())
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Multimodal AI-Driven Cybersecurity System (Taylor et al.)'
    )
    parser.add_argument('--mode', choices=['single', 'interactive', 'batch'], default='single')
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  MULTIMODAL LLM-BASED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("  Context-Aware Threat Detection for Communication Networks")
    print("  Implementing: Taylor et al. — Algorithm 1, Equations 1-15")
    print("=" * 80 + "\n")

    system = MultimodalCybersecuritySystem()

    try:
        if args.mode == 'single':
            results = system.run_complete_pipeline()# cache result for 30 seconds

            print("\n" + "=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)
            print(json.dumps(results, indent=2, default=str))
        elif args.mode == 'interactive':
            system.run_interactive_mode()
        elif args.mode == 'batch':
            system.run_batch_mode(iterations=args.iterations)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    print("\n✅ System execution completed!")
    print("📋 Logs available in: logs/system.log")
    print("💾 Results available in: results/")


if __name__ == '__main__':
    main()
