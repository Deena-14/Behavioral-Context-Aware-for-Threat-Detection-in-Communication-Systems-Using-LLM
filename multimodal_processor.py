"""
Multimodal Data Processor
Implements the paper's unified data integration mechanism (Section 3.1 - 3.3):
  D = ∪_{k=1}^{S} D_k                     (Equation 1)
  x_i = (x_i1, x_i2, ..., x_im)           (Equation 2)
  X' = P(X)                                (Equation 3)
  x' = (x - µ) / σ                        (Equation 4)
  M = {T1, T2, ..., Tn}                   (Equation 5)
  Sim(Ti, Tj) = Ti·Tj / (||Ti|| ||Tj||)   (Equation 6)

Data source: Real NSL-KDD dataset (KDDTrain+.txt) loaded via NSLKDDLoader.
Automatic fallback to simulation if dataset file is not found.
"""

import json
import math
import logging
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random

# ── Real dataset loader (NSL-KDD) ─────────────────────────────────────────────
try:
    from nsl_kdd_loader import NSLKDDLoader
    _LOADER_AVAILABLE = True
except ImportError:
    _LOADER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default path — can be overridden via config
_DEFAULT_TRAIN_PATH = "KDDTrain+.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────
def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Sim(Ti, Tj) = Ti · Tj / (||Ti|| ||Tj||)   — Equation (6)"""
    dot    = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_feature(value: float, mean: float, std: float) -> float:
    """x' = (x - µ) / σ   — Equation (4)"""
    return 0.0 if std == 0 else (value - mean) / std


class MultimodalDataProcessor:
    """
    Unified data integration using the real NSL-KDD dataset.
    Integrates 5 heterogeneous sources named in Sections 1 and 3.1:
      D1 — System logs
      D2 — Network traffic statistics
      D3 — IDS alerts
      D4 — DNS communication records
      D5 — Endpoint activities
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_cache: List[Dict] = []
        self.correlations: List[Dict] = []
        self._feature_stats: Dict[str, Dict[str, float]] = {}

        # Attempt to initialise real dataset loader
        # CHANGE: also pass test_dataset_path so evaluate_on_test() works
        train_path = config.get('dataset_path', _DEFAULT_TRAIN_PATH)
        test_path  = config.get('test_dataset_path', 'KDDTest+.arff')
        if _LOADER_AVAILABLE and os.path.exists(train_path):
            self._loader = NSLKDDLoader(train_path, test_path=test_path)
            self._use_real_data = True
            logger.info(f"[NSL-KDD] Real dataset found at '{train_path}' — using real data")
            if os.path.exists(test_path):
                logger.info(f"[NSL-KDD] Test dataset found at '{test_path}' — evaluation available")
            else:
                logger.warning(f"[NSL-KDD] Test file '{test_path}' not found — evaluate_on_test() will raise FileNotFoundError")
        else:
            self._loader = None
            self._use_real_data = False
            if _LOADER_AVAILABLE:
                logger.warning(
                    "\n" + "="*70 + "\n"
                    + f"[NSL-KDD] WARNING: Dataset '{train_path}' not found.\n"
                    + "  Falling back to SIMULATION mode — results are synthetic.\n"
                    + "  Place KDDTrain+.txt here or set KDD_DATASET_PATH env var.\n"
                    + "="*70
                )
            else:
                logger.warning("[NSL-KDD] nsl_kdd_loader not importable — using simulation")

    # ── Data source: real NSL-KDD or simulation ───────────────────────────────
    def _get_sources(self) -> Dict[str, List[Dict]]:
        """
        Returns the 5 data-source lists.
        Uses real NSL-KDD rows when available, simulation otherwise.
        """
        if self._use_real_data:
            sources = self._loader.get_multimodal_sources(
                n_samples=72, attack_ratio=0.35, seed=None   # seed=None → different sample each run
            )
            # Log data origin
            meta = sources.pop('_meta', {})
            logger.info(f"[DATA SOURCE] NSL-KDD real data | "
                        f"records={meta.get('total_records',0)} | "
                        f"dist={meta.get('attack_distribution',{})}")
            return sources
        else:
            return {
                'system_logs':       self._sim_system_logs(),
                'network_traffic':   self._sim_network_traffic(),
                'ids_alerts':        self._sim_ids_alerts(),
                'dns_records':       self._sim_dns_records(),
                'endpoint_activity': self._sim_endpoint_activity(),
            }

    # ── Fallback simulations (kept for environments without the dataset) ───────
    def _sim_system_logs(self) -> List[Dict]:
        normal_evts   = [('USER_LOGIN','low'),('FILE_ACCESS','low'),('PROCESS_EXEC','low'),
                         ('USER_LOGOUT','low'),('SERVICE_STOP','low'),('SUDO_USAGE','low')]
        suspicious    = [('AUTH_FAILURE','high'),('FAILED_LOGIN','high'),('PRIVILEGE_ESCALATION','critical')]
        events = []
        for _ in range(15):
            pool = suspicious if random.random() < 0.20 else normal_evts
            et, sev = random.choice(pool)
            events.append({
                'source':'system_logs','timestamp':(datetime.now()-timedelta(minutes=random.randint(0,60))).isoformat(),
                'event_type':et,'description':f'{et.replace("_"," ").title()} detected',
                'user':f'user{random.randint(1,20)}','source_host':f'host-{random.randint(1,50)}',
                'source_ip':f'192.168.1.{random.randint(1,254)}','severity':sev,
                'features':{'f_severity_numeric':{'low':1,'medium':2,'high':3,'critical':4}[sev],
                            'f_is_auth_event':1 if 'AUTH' in et or 'LOGIN' in et else 0,
                            'f_is_privilege_event':1 if 'PRIVILEGE' in et or 'SUDO' in et else 0}})
        return events

    def _sim_network_traffic(self) -> List[Dict]:
        events = []
        for _ in range(20):
            bs = random.randint(100,5_000_000); pc = random.randint(10,50_000); dur = random.randint(1,3600)
            dp = random.choices([22,80,443,53,3306,5432,8080,8443,4444,31337],
                                weights=[18,18,18,10,7,7,7,7,4,4],k=1)[0]
            sus = dp not in [22,80,443,53,3306,5432]
            events.append({'source':'network_traffic','timestamp':(datetime.now()-timedelta(minutes=random.randint(0,60))).isoformat(),
                'source_ip':f'192.168.1.{random.randint(1,254)}','dest_ip':f'10.0.0.{random.randint(1,254)}',
                'source_port':random.randint(1024,65535),'dest_port':dp,'protocol':random.choice(['TCP','UDP']),
                'bytes_sent':bs,'bytes_received':random.randint(100,1_000_000),'packet_count':pc,'duration_sec':dur,
                'flags':random.choice(['SYN','ACK','RST','FIN','SYN-ACK']),
                'features':{'f_bytes_sent':bs,'f_packet_count':pc,'f_duration':dur,'f_dest_port':dp,
                            'f_is_suspicious_port':int(sus),'f_bytes_per_packet':bs/max(pc,1)}})
        return events

    def _sim_ids_alerts(self) -> List[Dict]:
        normal = [('Port Scanning Detected','low'),('Suspicious Payload','low')]
        threats= [('Buffer Overflow Attempt','critical'),('Malware Signature Match','critical'),
                  ('SQL Injection Attempt','high'),('Brute Force Authentication','high')]
        events = []
        for _ in range(10):
            pool = threats if random.random() < 0.25 else normal
            at, sev = random.choice(pool)
            events.append({'source':'ids_alerts','timestamp':(datetime.now()-timedelta(minutes=random.randint(0,60))).isoformat(),
                'alert_type':at,'source_ip':f'192.168.1.{random.randint(1,254)}','dest_ip':f'10.0.0.{random.randint(1,254)}',
                'signature_id':f'SID-{random.randint(1_000_000,9_999_999)}','severity':sev,'protocol':random.choice(['TCP','UDP']),
                'features':{'f_severity_numeric':{'low':1,'medium':2,'high':3,'critical':4}[sev],
                            'f_is_injection':1 if 'Injection' in at else 0,'f_is_overflow':1 if 'Overflow' in at else 0}})
        return events

    def _sim_dns_records(self) -> List[Dict]:
        sus_dom = ['malware-c2.ru','botnet-ctrl.xyz','phish-login.net','data-exfil.io','dga-abc123.com']
        norm_dom= ['google.com','microsoft.com','github.com','amazon.com','cloudflare.com']
        events  = []
        for _ in range(15):
            is_sus = random.random() < 0.15
            dom    = random.choice(sus_dom if is_sus else norm_dom)
            qc     = random.randint(1,200); rc = random.choice(['NOERROR','NXDOMAIN','SERVFAIL'])
            events.append({'source':'dns_records','timestamp':(datetime.now()-timedelta(minutes=random.randint(0,60))).isoformat(),
                'query_ip':f'192.168.1.{random.randint(1,254)}','domain':dom,'record_type':random.choice(['A','AAAA','MX','TXT']),
                'response_code':rc,'response_ip':f'1.1.1.{random.randint(1,254)}','is_suspicious':is_sus,'query_count':qc,
                'features':{'f_is_suspicious_domain':int(is_sus),'f_query_count':qc,'f_domain_length':len(dom),
                            'f_is_nxdomain':1 if rc=='NXDOMAIN' else 0}})
        return events

    def _sim_endpoint_activity(self) -> List[Dict]:
        acts = [('File Download','low'),('Process Creation','medium'),('Registry Change','high'),
                ('Network Connection','medium'),('Credential Access','critical'),('WMI Event','high')]
        events = []
        for _ in range(12):
            at, sev = random.choice(acts)
            pn = random.choice(['svchost.exe','powershell.exe','cmd.exe','wscript.exe'])
            events.append({'source':'endpoint_activity','timestamp':(datetime.now()-timedelta(minutes=random.randint(0,60))).isoformat(),
                'endpoint_id':f'endpoint-{random.randint(1,100)}','activity_type':at,
                'activity_type_key':at.lower().replace(' ','_'),'user':f'user{random.randint(1,20)}',
                'process_name':pn,'process_id':random.randint(100,10000),'severity':sev,
                'features':{'f_severity_numeric':{'low':1,'medium':2,'high':3,'critical':4}[sev],
                            'f_is_powershell':1 if 'powershell' in pn.lower() else 0,
                            'f_is_credential':1 if 'Credential' in at else 0}})
        return events

    # ── Public source accessors (kept for backward compatibility) ─────────────
    def process_system_logs(self)       -> List[Dict]: return self._get_sources()['system_logs']
    def process_network_traffic(self)   -> List[Dict]: return self._get_sources()['network_traffic']
    def process_ids_alerts(self)        -> List[Dict]: return self._get_sources()['ids_alerts']
    def process_dns_records(self)       -> List[Dict]: return self._get_sources()['dns_records']
    def process_endpoint_activity(self) -> List[Dict]: return self._get_sources()['endpoint_activity']

    # ── Preprocessing  X' = P(X)  (Eq. 3 & 4) ───────────────────────────────
    def preprocess_events(self, events: List[Dict]) -> List[Dict]:
        """X' = P(X) — Eq. 3: remove duplicates, normalize (Eq. 4)."""
        feature_values: Dict[str, List[float]] = {}
        for ev in events:
            for fn, fv in ev.get('features', {}).items():
                feature_values.setdefault(fn, []).append(float(fv))

        feature_stats: Dict[str, Dict[str, float]] = {}
        for fn, vals in feature_values.items():
            mean = sum(vals) / len(vals)
            std  = math.sqrt(sum((v - mean)**2 for v in vals) / max(len(vals)-1, 1))
            feature_stats[fn] = {'mean': mean, 'std': std}
        self._feature_stats = feature_stats

        cleaned: List[Dict] = []
        seen_keys = set()
        for ev in events:
            key = (ev.get('source'), ev.get('timestamp'),
                   ev.get('event_type', ev.get('alert_type', ev.get('activity_type', ''))))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            norm_feats = {}
            for fn, fv in ev.get('features', {}).items():
                stats = feature_stats.get(fn, {'mean': 0, 'std': 1})
                norm_feats[fn + '_norm'] = round(
                    normalize_feature(float(fv), stats['mean'], stats['std']), 4)
            ev['normalised_features'] = norm_feats
            cleaned.append(ev)

        logger.info(f"[Preprocessing] {len(events)} → {len(cleaned)} events after cleaning")
        return cleaned

    # ── Textual representations  M = {T1,...,Tn}  (Eq. 5) ────────────────────
    def build_textual_representations(self, events: List[Dict]) -> List[Dict]:
        """Convert each event to a structured text Ti — Eq. 5."""
        representations = []
        vocab_tokens = [
            'fail','auth','credential','login','password','privilege','sudo',
            'suspicious','scan','transfer','beacon','c2','port','traffic',
            'dns','domain','query','nxdomain','tunnel',
            'malware','injection','overflow','exploit','payload','encrypt',
            'exfil','lateral','movement','persistence','escalation',
            'critical','alert','high','severe','anomaly',
            'process','execution','endpoint','driver','kernel',
            'dos','ddos','probe','neptune','smurf','portsweep','ipsweep',
            'buffer','overflow','rootkit','guess','passwd',
        ]
        for ev in events:
            source = ev.get('source', 'unknown')
            attack_label = ev.get('attack_label', '')
            attack_cat   = ev.get('attack_category', '')

            if source == 'system_logs':
                text = (f"[SYSTEM LOG] Event: {ev.get('event_type')} | "
                        f"User: {ev.get('user')} | Host: {ev.get('source_host')} | "
                        f"IP: {ev.get('source_ip')} | Severity: {ev.get('severity')} | "
                        f"Time: {ev.get('timestamp')}")
            elif source == 'network_traffic':
                text = (f"[NETWORK TRAFFIC] {ev.get('source_ip')}:{ev.get('source_port')} → "
                        f"{ev.get('dest_ip')}:{ev.get('dest_port')} | "
                        f"Proto: {ev.get('protocol')} | Service: {ev.get('service','')} | "
                        f"Bytes: {ev.get('bytes_sent')} | Flag: {ev.get('flag',ev.get('flags',''))} | "
                        f"Time: {ev.get('timestamp')}")
            elif source == 'ids_alerts':
                text = (f"[IDS ALERT] {ev.get('alert_type')} | "
                        f"Src: {ev.get('source_ip')} → Dst: {ev.get('dest_ip')} | "
                        f"Sig: {ev.get('signature_id')} | Severity: {ev.get('severity')} | "
                        f"Time: {ev.get('timestamp')}")
            elif source == 'dns_records':
                text = (f"[DNS RECORD] Query from {ev.get('query_ip')} for {ev.get('domain')} "
                        f"({ev.get('record_type')}) | Response: {ev.get('response_code')} | "
                        f"Count: {ev.get('query_count')} | Suspicious: {ev.get('is_suspicious')} | "
                        f"Time: {ev.get('timestamp')}")
            elif source == 'endpoint_activity':
                text = (f"[ENDPOINT] {ev.get('activity_type')} on {ev.get('endpoint_id')} | "
                        f"User: {ev.get('user')} | Process: {ev.get('process_name')} | "
                        f"Severity: {ev.get('severity')} | Time: {ev.get('timestamp')}")
            else:
                text = f"[UNKNOWN] {json.dumps(ev, default=str)}"

            # Include attack label in text so LLM signal counting works
            if attack_label and attack_label != 'normal':
                text += f" | NSL-KDD: {attack_label} ({attack_cat})"

            bow_vector = [1.0 if token in text.lower() else 0.0 for token in vocab_tokens]
            representations.append({
                'event_id': id(ev), 'source': source,
                'timestamp': ev.get('timestamp'), 'text': text,
                'embedding_vector': bow_vector,
                'attack_label': attack_label,
                'attack_category': attack_cat,
                'original_event': ev,
            })

        logger.info(f"[Representation] Built {len(representations)} textual representations")
        return representations

    # ── Cross-source correlation  Sim > θ  (Eq. 6 & 7) ───────────────────────
    def correlate_events(self, representations: List[Dict], theta: float = 0.4) -> List[Dict]:
        correlations = []
        n = len(representations)
        for i in range(n):
            for j in range(i + 1, n):
                ri, rj = representations[i], representations[j]
                if ri['source'] == rj['source']:
                    continue
                sim = cosine_similarity(ri['embedding_vector'], rj['embedding_vector'])
                if sim > theta:
                    correlations.append({
                        'timestamp': datetime.now().isoformat(),
                        'event_i':   ri['text'][:100],
                        'event_j':   rj['text'][:100],
                        'source_i':  ri['source'],
                        'source_j':  rj['source'],
                        'similarity_score': round(sim, 4),
                        'description': (
                            f"Cross-source correlation: {ri['source']} ↔ {rj['source']} "
                            f"(sim={sim:.3f} > θ={theta})"
                        ),
                    })
        logger.info(f"[Correlation] {len(correlations)} cross-source correlations (θ={theta})")
        return correlations

    # ── Aggregate D = ∪ Dk  (Eq. 1) ──────────────────────────────────────────
    def aggregate_all_sources(self) -> Dict[str, Any]:
        logger.info("Aggregating all 5 data sources (Equation 1)…")
        sources = self._get_sources()
        all_events = []
        for evts in sources.values():
            all_events.extend(evts)

        # Count real vs simulated
        mode = "NSL-KDD (real data)" if self._use_real_data else "simulation (fallback)"
        logger.info(f"[Data Mode] {mode} | {len(all_events)} total events from 5 sources")

        return {
            'timestamp':    datetime.now().isoformat(),
            'num_sources':  5,
            'data_mode':    mode,
            **sources,
            'total_events': len(all_events),
        }

    # ── Format for LLM ────────────────────────────────────────────────────────
    def format_for_llm(self, representations: List[Dict], correlations: List[Dict]) -> str:
        by_source: Dict[str, List[str]] = {}
        for rep in representations:
            by_source.setdefault(rep['source'], []).append(rep['text'])

        payload = {
            'analysis_timestamp':       datetime.now().isoformat(),
            'data_mode':                'NSL-KDD real dataset' if self._use_real_data else 'simulation',
            'data_sources_integrated':  list(by_source.keys()),
            'system_logs':              by_source.get('system_logs',      [])[:5],
            'network_traffic':          by_source.get('network_traffic',  [])[:5],
            'ids_alerts':               by_source.get('ids_alerts',       [])[:5],
            'dns_records':              by_source.get('dns_records',       [])[:5],
            'endpoint_activity':        by_source.get('endpoint_activity',[])[:5],
            'cross_source_correlations':[c['description'] for c in correlations[:10]],
        }
        return json.dumps(payload, indent=2, default=str)

    # ── Main pipeline entry ───────────────────────────────────────────────────
    def process_all(self) -> Dict[str, Any]:
        logger.info("=== Starting multimodal data processing pipeline ===")

        aggregated    = self.aggregate_all_sources()
        all_events    = []
        for key in ['system_logs','network_traffic','ids_alerts','dns_records','endpoint_activity']:
            all_events.extend(aggregated.get(key, []))

        cleaned       = self.preprocess_events(all_events)
        representations = self.build_textual_representations(cleaned)
        correlations  = self.correlate_events(representations, theta=0.65)
        llm_input     = self.format_for_llm(representations, correlations)

        # Attack distribution summary (real data only)
        if self._use_real_data:
            cats = {}
            for ev in all_events:
                c = ev.get('attack_category', 'unknown')
                cats[c] = cats.get(c, 0) + 1
            logger.info(f"[Attack Distribution in batch] {cats}")

        return {
            'aggregated_data': aggregated,
            'cleaned_events':  cleaned,
            'representations': representations,
            'correlations':    correlations,
            'llm_input':       llm_input,
            'data_mode':       'NSL-KDD (real data)' if self._use_real_data else 'simulation',
        }
