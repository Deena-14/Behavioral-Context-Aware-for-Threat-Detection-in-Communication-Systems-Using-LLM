"""
nsl_kdd_loader.py
Loads and preprocesses the real NSL-KDD dataset (KDDTrain+.txt / KDDTest+.arff)
and maps it to the 5 multimodal data sources used in the paper:
  D1 — System Logs
  D2 — Network Traffic Statistics
  D3 — IDS Alerts
  D4 — DNS Communication Records
  D5 — Endpoint Activities

CHANGES (v2):
  - evaluate_on_test(): loads KDDTest+.arff, runs predictions through the
    pipeline, and returns accuracy / per-category metrics vs ground truth.
    Previously test_path was accepted but never used.
  - _load_arff(): parses the ARFF format directly (no scipy dependency).
  - ARFF labels are 'normal'/'anomaly' — mapped to binary is_attack for eval.

Paper reference: Section 3.1, Section 4.1, Table 2
"""

import os
import math
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Column names (from notebook cell 6) ──────────────────────────────────────
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'level',
]

# ── Attack → category mapping (paper Table 2: Probe/DoS/R2L/U2R) ─────────────
ATTACK_CATEGORY_MAP = {
    'normal':           'normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
    'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    # Probe
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
    'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'warezclient': 'R2L', 'warezmaster': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'named': 'R2L', 'sendmail': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
    'sqlattack': 'U2R', 'xterm': 'U2R',
}

CATEGORY_TO_THREAT = {
    'DoS':    'ddos_attack',
    'Probe':  'port_scanning',
    'R2L':    'credential_stuffing',
    'U2R':    'privilege_escalation',
    'normal': 'no_threat',
}

SERVICE_PORT_MAP = {
    'http': 80, 'https': 443, 'ftp': 21, 'ftp_data': 20,
    'smtp': 25, 'ssh': 22, 'telnet': 23, 'dns': 53, 'domain_u': 53,
    'pop_3': 110, 'imap4': 143, 'finger': 79, 'sunrpc': 111,
    'auth': 113, 'bgp': 179, 'ldap': 389, 'private': 1024,
    'other': 9999, 'eco_i': 7, 'ecr_i': 7,
}


class NSLKDDLoader:
    """
    Loads the real NSL-KDD dataset and maps each record to the
    paper's 5 heterogeneous data sources (Section 3.1).

    Usage:
        loader = NSLKDDLoader("KDDTrain+.txt", test_path="KDDTest+.arff")
        sources = loader.get_multimodal_sources(n_samples=50)
        report  = loader.evaluate_on_test(analyzer)   # NEW
    """

    def __init__(self, train_path: str = "KDDTrain+.txt",
                 test_path: str = None):
        self.train_path = train_path
        self.test_path  = test_path
        self._df_train: pd.DataFrame = None
        self._df_test:  pd.DataFrame = None
        self._loaded = False

    # ── Load & preprocess training data ──────────────────────────────────────
    def _load(self):
        if self._loaded:
            return
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(
                f"NSL-KDD training file not found: {self.train_path}\n"
                "Download KDDTrain+.txt from Kaggle and place it in the project folder."
            )
        logger.info(f"Loading NSL-KDD dataset from {self.train_path}…")
        df = pd.read_csv(self.train_path, names=NSL_KDD_COLUMNS, header=None)
        df['attack_category'] = df['attack'].map(ATTACK_CATEGORY_MAP).fillna('Other')
        df['threat_type']     = df['attack_category'].map(CATEGORY_TO_THREAT).fillna('no_threat')
        df['is_attack']       = (df['attack'] != 'normal').astype(int)
        self._df_train = df
        self._loaded   = True
        logger.info(f"Loaded {len(df):,} records | "
                    f"Normal: {(df['attack']=='normal').sum():,} | "
                    f"Attack: {(df['attack']!='normal').sum():,}")

    # ── CHANGE: Load KDDTest+.arff (was previously unused) ───────────────────
    def _load_arff(self) -> pd.DataFrame:
        """
        Parse KDDTest+.arff without scipy.
        The ARFF has 41 numeric/categorical attributes + 'class' label
        ('normal' or 'anomaly') — no specific attack name is given.
        We map: normal → is_attack=0, anomaly → is_attack=1.
        """
        if self._df_test is not None:
            return self._df_test

        if not self.test_path or not os.path.exists(self.test_path):
            raise FileNotFoundError(
                f"KDDTest+.arff not found at '{self.test_path}'. "
                "Place KDDTest+.arff in the project folder."
            )

        logger.info(f"Loading KDDTest+.arff from {self.test_path}…")

        rows = []
        in_data = False
        with open(self.test_path, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                if line.lower() == '@data':
                    in_data = True
                    continue
                if in_data:
                    rows.append(line.split(','))

        # The ARFF has 41 feature columns + 1 label column (no 'level')
        arff_cols = NSL_KDD_COLUMNS[:41] + ['attack']
        df = pd.DataFrame(rows, columns=arff_cols)

        # Strip whitespace and quotes from string columns
        for col in ['protocol_type', 'service', 'flag', 'attack']:
            df[col] = df[col].str.strip().str.strip("'\"")

        # Convert numeric columns
        num_cols = [c for c in arff_cols if c not in ('protocol_type', 'service', 'flag', 'attack')]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # ARFF label is 'normal' or 'anomaly' (binary), not specific attack names
        df['is_attack'] = (df['attack'].str.lower() == 'anomaly').astype(int)
        df['attack_category'] = df['attack'].apply(
            lambda x: 'normal' if x.lower() == 'normal' else 'Unknown'
        )
        df['threat_type'] = df['is_attack'].apply(
            lambda x: 'no_threat' if x == 0 else 'unknown_attack'
        )
        df['level'] = 0   # not present in ARFF

        self._df_test = df
        logger.info(f"KDDTest+.arff loaded: {len(df):,} records | "
                    f"Normal: {(df['is_attack']==0).sum():,} | "
                    f"Anomaly: {(df['is_attack']==1).sum():,}")
        return df

    def get_dataframe(self) -> pd.DataFrame:
        self._load()
        return self._df_train.copy()

    def get_sample(self, n: int = 100, include_attacks: bool = True,
                   attack_ratio: float = 0.3, seed: int = None) -> pd.DataFrame:
        """Return a stratified sample of n records."""
        self._load()
        df = self._df_train
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n_attack = int(n * attack_ratio) if include_attacks else 0
        n_normal = n - n_attack

        normal_idx = df[df['attack'] == 'normal'].sample(
            min(n_normal, (df['attack'] == 'normal').sum()), random_state=seed).index
        if n_attack > 0:
            attack_idx = df[df['attack'] != 'normal'].sample(
                min(n_attack, (df['attack'] != 'normal').sum()), random_state=seed).index
            idx = list(normal_idx) + list(attack_idx)
        else:
            idx = list(normal_idx)

        return df.loc[idx].reset_index(drop=True)

    # ── Fake IP / timestamp helpers ───────────────────────────────────────────
    @staticmethod
    def _make_src_ip(i: int) -> str:
        return f"192.168.{(i // 254) % 254 + 1}.{i % 254 + 1}"

    @staticmethod
    def _make_dst_ip(i: int) -> str:
        return f"10.0.{(i // 254) % 254 + 1}.{i % 254 + 1}"

    @staticmethod
    def _make_ts(offset_min: int = 0) -> str:
        return (datetime.now() - timedelta(minutes=offset_min)).isoformat()

    # ── Row mappers (unchanged from v1) ──────────────────────────────────────
    def _row_to_system_log(self, row: pd.Series, idx: int) -> Dict:
        attack_cat = row['attack_category']
        is_attack  = row['is_attack']

        if attack_cat == 'U2R':
            event_type = 'PRIVILEGE_ESCALATION'; severity = 'critical'
        elif row['num_failed_logins'] > 0 or attack_cat == 'R2L':
            event_type = 'FAILED_LOGIN' if row['num_failed_logins'] > 0 else 'AUTH_FAILURE'
            severity = 'high'
        elif row['root_shell'] == 1:
            event_type = 'PRIVILEGE_ESCALATION'; severity = 'critical'
        elif row['su_attempted'] == 1:
            event_type = 'SUDO_USAGE'; severity = 'high'
        elif is_attack:
            event_type = 'SUSPICIOUS_ACTIVITY'; severity = 'medium'
        else:
            event_type = random.choice(['USER_LOGIN', 'FILE_ACCESS', 'PROCESS_EXEC', 'USER_LOGOUT'])
            severity = 'low'

        return {
            'source': 'system_logs', 'timestamp': self._make_ts(idx),
            'event_type': event_type,
            'description': f"{event_type.replace('_', ' ').title()} detected",
            'user': f"user{(idx % 20) + 1}", 'source_host': f"host-{(idx % 50) + 1}",
            'source_ip': self._make_src_ip(idx), 'severity': severity,
            'attack_label': row['attack'], 'attack_category': attack_cat,
            'features': {
                'f_severity_numeric':   {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}[severity],
                'f_is_auth_event':      1 if 'AUTH' in event_type or 'LOGIN' in event_type else 0,
                'f_is_privilege_event': 1 if 'PRIVILEGE' in event_type or 'SUDO' in event_type else 0,
                'f_num_failed_logins':  float(row['num_failed_logins']),
                'f_root_shell':         float(row['root_shell']),
                'f_logged_in':          float(row['logged_in']),
            },
        }

    def _row_to_network_traffic(self, row: pd.Series, idx: int) -> Dict:
        port   = SERVICE_PORT_MAP.get(row['service'], 9999)
        is_sus = row['is_attack'] == 1

        return {
            'source': 'network_traffic', 'timestamp': self._make_ts(idx),
            'source_ip': self._make_src_ip(idx), 'dest_ip': self._make_dst_ip(idx),
            'source_port': random.randint(1024, 65535), 'dest_port': port,
            'protocol': str(row['protocol_type']).upper(), 'service': row['service'],
            'flag': row['flag'],
            'bytes_sent': int(row['src_bytes']), 'bytes_received': int(row['dst_bytes']),
            'duration_sec': int(row['duration']), 'packet_count': int(row['count']),
            'serror_rate': float(row['serror_rate']), 'rerror_rate': float(row['rerror_rate']),
            'attack_label': row['attack'], 'attack_category': row['attack_category'],
            'features': {
                'f_src_bytes':      float(row['src_bytes']),
                'f_dst_bytes':      float(row['dst_bytes']),
                'f_duration':       float(row['duration']),
                'f_count':          float(row['count']),
                'f_srv_count':      float(row['srv_count']),
                'f_serror_rate':    float(row['serror_rate']),
                'f_rerror_rate':    float(row['rerror_rate']),
                'f_same_srv_rate':  float(row['same_srv_rate']),
                'f_diff_srv_rate':  float(row['diff_srv_rate']),
                'f_dest_port':      float(port),
                'f_is_suspicious':  float(is_sus),
            },
        }

    def _row_to_ids_alert(self, row: pd.Series, idx: int) -> Dict:
        cat    = row['attack_category']
        attack = row['attack']

        alert_map = {
            'DoS':    ('DoS / DDoS Attack Detected',       'critical'),
            'Probe':  ('Port Scanning / Reconnaissance',   'high'),
            'R2L':    ('Remote to Local Attack Attempt',   'high'),
            'U2R':    ('User to Root Privilege Escalation','critical'),
            'normal': (random.choice(['Low Severity Noise', 'Suspicious Payload',
                                      'Port Scanning Detected']), 'low'),
        }
        alert_type, severity = alert_map.get(cat, ('Unknown Alert', 'medium'))

        return {
            'source': 'ids_alerts', 'timestamp': self._make_ts(idx),
            'alert_type': alert_type,
            'source_ip': self._make_src_ip(idx), 'dest_ip': self._make_dst_ip(idx),
            'signature_id': f"SID-{abs(hash(attack)) % 9_000_000 + 1_000_000}",
            'severity': severity, 'protocol': str(row['protocol_type']).upper(),
            'attack_label': attack, 'attack_category': cat,
            'features': {
                'f_severity_numeric': {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}[severity],
                'f_is_dos':           1 if cat == 'DoS'   else 0,
                'f_is_probe':         1 if cat == 'Probe' else 0,
                'f_is_r2l':           1 if cat == 'R2L'   else 0,
                'f_is_u2r':           1 if cat == 'U2R'   else 0,
                'f_wrong_fragment':   float(row['wrong_fragment']),
                'f_urgent':           float(row['urgent']),
            },
        }

    def _row_to_dns_record(self, row: pd.Series, idx: int) -> Dict:
        cat        = row['attack_category']
        is_sus     = cat in ('R2L', 'U2R', 'Probe')
        is_dos_dns = (cat == 'DoS' and row['service'] in ('domain_u', 'dns'))

        if is_sus:
            domain = random.choice(['malware-c2.ru', 'botnet-ctrl.xyz', 'phish-login.net',
                                    'data-exfil.io', 'dga-abc123.com'])
        elif is_dos_dns:
            domain = random.choice(['flood-target.net', 'amplify-dns.org'])
        else:
            domain = random.choice(['google.com', 'microsoft.com', 'github.com',
                                    'amazon.com', 'cloudflare.com'])

        query_count   = int(row['dst_host_count'])
        response_code = 'NXDOMAIN' if (row['rerror_rate'] > 0.5) else 'NOERROR'

        return {
            'source': 'dns_records', 'timestamp': self._make_ts(idx),
            'query_ip': self._make_src_ip(idx), 'domain': domain,
            'record_type': random.choice(['A', 'AAAA', 'MX', 'TXT', 'CNAME']),
            'response_code': response_code,
            'response_ip': f"1.1.1.{idx % 254 + 1}",
            'is_suspicious': bool(is_sus or is_dos_dns),
            'query_count': query_count,
            'attack_label': row['attack'], 'attack_category': cat,
            'features': {
                'f_is_suspicious_domain':   int(is_sus or is_dos_dns),
                'f_query_count':            float(query_count),
                'f_domain_length':          float(len(domain)),
                'f_is_nxdomain':            1 if response_code == 'NXDOMAIN' else 0,
                'f_dst_host_srv_count':     float(row['dst_host_srv_count']),
                'f_dst_host_serror_rate':   float(row['dst_host_serror_rate']),
            },
        }

    def _row_to_endpoint_activity(self, row: pd.Series, idx: int) -> Dict:
        cat = row['attack_category']

        if cat == 'U2R' or row['root_shell'] == 1:
            activity_type = 'Credential Access'; severity = 'critical'
        elif cat == 'R2L':
            activity_type = 'Lateral Movement'; severity = 'high'
        elif row['num_file_creations'] > 0:
            activity_type = 'File Download'; severity = 'medium'
        elif row['num_shells'] > 0:
            activity_type = 'Process Creation'; severity = 'high'
        elif cat == 'Probe':
            activity_type = 'Network Connection'; severity = 'medium'
        elif cat == 'DoS':
            activity_type = 'Service Install'; severity = 'high'
        else:
            activity_type = random.choice(['File Download', 'Process Creation', 'Network Connection'])
            severity = 'low'

        process_name  = random.choice(['svchost.exe', 'powershell.exe', 'cmd.exe', 'wscript.exe'])
        is_powershell = 'powershell' in process_name.lower()

        return {
            'source': 'endpoint_activity', 'timestamp': self._make_ts(idx),
            'endpoint_id': f"endpoint-{(idx % 100) + 1}",
            'activity_type': activity_type,
            'activity_type_key': activity_type.lower().replace(' ', '_'),
            'user': f"user{(idx % 20) + 1}",
            'process_name': process_name, 'process_id': random.randint(100, 10000),
            'file_path': f"C:\\Users\\User{(idx % 5) + 1}\\AppData\\...",
            'command_line': 'Encoded command execution detected' if cat != 'normal' else 'Normal execution',
            'severity': severity,
            'attack_label': row['attack'], 'attack_category': cat,
            'features': {
                'f_severity_numeric':   {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}[severity],
                'f_is_powershell':      int(is_powershell),
                'f_num_file_creations': float(row['num_file_creations']),
                'f_num_shells':         float(row['num_shells']),
                'f_is_credential':      1 if 'Credential' in activity_type else 0,
                'f_num_compromised':    float(row['num_compromised']),
                'f_hot':                float(row['hot']),
            },
        }

    # ── Main: get all 5 sources from real training data ───────────────────────
    def get_multimodal_sources(self, n_samples: int = 72,
                                attack_ratio: float = 0.35,
                                seed: int = 42) -> Dict[str, List[Dict]]:
        self._load()
        sample = self.get_sample(n_samples, attack_ratio=attack_ratio, seed=seed)

        n   = len(sample)
        idx = list(range(n))
        random.shuffle(idx)

        cuts = [0,
                int(n * 15 / 72),
                int(n * 35 / 72),
                int(n * 49 / 72),
                int(n * 63 / 72),
                n]

        log_rows = sample.iloc[idx[cuts[0]:cuts[1]]].reset_index(drop=True)
        net_rows = sample.iloc[idx[cuts[1]:cuts[2]]].reset_index(drop=True)
        ids_rows = sample.iloc[idx[cuts[2]:cuts[3]]].reset_index(drop=True)
        dns_rows = sample.iloc[idx[cuts[3]:cuts[4]]].reset_index(drop=True)
        ep_rows  = sample.iloc[idx[cuts[4]:cuts[5]]].reset_index(drop=True)

        system_logs       = [self._row_to_system_log(r, i)       for i, r in log_rows.iterrows()]
        network_traffic   = [self._row_to_network_traffic(r, i)  for i, r in net_rows.iterrows()]
        ids_alerts        = [self._row_to_ids_alert(r, i)        for i, r in ids_rows.iterrows()]
        dns_records       = [self._row_to_dns_record(r, i)       for i, r in dns_rows.iterrows()]
        endpoint_activity = [self._row_to_endpoint_activity(r, i) for i, r in ep_rows.iterrows()]

        all_rows = pd.concat([log_rows, net_rows, ids_rows, dns_rows, ep_rows])
        attack_counts = all_rows['attack_category'].value_counts().to_dict()

        logger.info(f"[NSL-KDD] {n} real records loaded → "
                    f"Logs:{len(system_logs)} Net:{len(network_traffic)} "
                    f"IDS:{len(ids_alerts)} DNS:{len(dns_records)} "
                    f"Endpoint:{len(endpoint_activity)}")
        logger.info(f"[NSL-KDD] Attack distribution: {attack_counts}")

        return {
            'system_logs':       system_logs,
            'network_traffic':   network_traffic,
            'ids_alerts':        ids_alerts,
            'dns_records':       dns_records,
            'endpoint_activity': endpoint_activity,
            '_meta': {
                'source':             'NSL-KDD (KDDTrain+.txt)',
                'total_records':      n,
                'attack_distribution': attack_counts,
                'columns':            NSL_KDD_COLUMNS,
            }
        }

    # ── CHANGE: evaluate_on_test() — previously unused test_path now used ─────
    def evaluate_on_test(self, analyzer, n_samples: int = 200,
                         seed: int = 42) -> Dict[str, Any]:
        """
        Load KDDTest+.arff, run each record through the signal-counting
        pipeline, and compare predicted is_attack against ground truth.

        Returns a report dict with accuracy, precision, recall, F1,
        and per-category breakdown.

        NOTE: The ARFF only has binary labels (normal / anomaly), so
        evaluation is binary (attack detected vs not detected).

        Parameters
        ----------
        analyzer : LLMThreatAnalyzer
            An initialised LLMThreatAnalyzer used for signal counting.
        n_samples : int
            How many test records to evaluate (full set = ~22 544 rows).
        seed : int
            Random seed for reproducible sampling.
        """
        import json as _json

        df_test = self._load_arff()

        # Stratified sample from test set
        np.random.seed(seed)
        normal_idx = df_test[df_test['is_attack'] == 0].index.tolist()
        attack_idx = df_test[df_test['is_attack'] == 1].index.tolist()

        n_attack = min(int(n_samples * 0.45), len(attack_idx))
        n_normal = min(n_samples - n_attack, len(normal_idx))

        sampled_idx = (
            list(np.random.choice(normal_idx, n_normal, replace=False))
            + list(np.random.choice(attack_idx, n_attack, replace=False))
        )
        sample = df_test.loc[sampled_idx].reset_index(drop=True)

        logger.info(f"[Evaluation] Running on {len(sample)} test records "
                    f"(normal={n_normal}, attack={n_attack})…")

        y_true, y_pred = [], []
        category_results: Dict[str, Dict[str, int]] = {}

        for i, row in sample.iterrows():
            # Build a minimal payload mimicking the LLM input for this single row
            net_ev = self._row_to_network_traffic(row, i)
            log_ev = self._row_to_system_log(row, i)
            ids_ev = self._row_to_ids_alert(row, i)
            dns_ev = self._row_to_dns_record(row, i)
            ep_ev  = self._row_to_endpoint_activity(row, i)

            payload = _json.dumps({
                'system_logs':              [log_ev],
                'network_traffic':          [net_ev],
                'ids_alerts':               [ids_ev],
                'dns_records':              [dns_ev],
                'endpoint_activity':        [ep_ev],
                'cross_source_correlations': [],
            })

            signals, _ = analyzer._count_suspicion_signals(payload)
            predicted_attack = 1 if signals >= 1 else 0

            true_label = int(row['is_attack'])
            y_true.append(true_label)
            y_pred.append(predicted_attack)

            # Track per-category (binary: normal vs anomaly)
            cat = 'normal' if true_label == 0 else 'anomaly'
            if cat not in category_results:
                category_results[cat] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
            if true_label == 1 and predicted_attack == 1:
                category_results[cat]['tp'] += 1
            elif true_label == 0 and predicted_attack == 1:
                category_results[cat]['fp'] += 1
            elif true_label == 0 and predicted_attack == 0:
                category_results[cat]['tn'] += 1
            else:
                category_results[cat]['fn'] += 1

        # ── Aggregate metrics ─────────────────────────────────────────────────
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        accuracy  = (tp + tn) / max(len(y_true), 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        fpr       = fp / max(fp + tn, 1)   # false positive rate

        report = {
            'test_file':    self.test_path,
            'n_evaluated':  len(y_true),
            'n_normal':     n_normal,
            'n_attack':     n_attack,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'accuracy':   round(accuracy,   4),
            'precision':  round(precision,  4),
            'recall':     round(recall,     4),
            'f1_score':   round(f1,         4),
            'false_positive_rate': round(fpr, 4),
            'category_breakdown': category_results,
        }

        logger.info(
            f"[Evaluation] Results — "
            f"Accuracy={accuracy:.3f} Precision={precision:.3f} "
            f"Recall={recall:.3f} F1={f1:.3f} FPR={fpr:.3f}"
        )
        return report

    # ── Dataset statistics ────────────────────────────────────────────────────
    def get_dataset_stats(self) -> Dict[str, Any]:
        self._load()
        df = self._df_train
        return {
            'total_records':     len(df),
            'normal_count':      int((df['attack'] == 'normal').sum()),
            'attack_count':      int((df['attack'] != 'normal').sum()),
            'attack_types':      df['attack'].value_counts().to_dict(),
            'attack_categories': df['attack_category'].value_counts().to_dict(),
            'protocol_dist':     df['protocol_type'].value_counts().to_dict(),
            'top_services':      df['service'].value_counts().head(10).to_dict(),
            'columns':           NSL_KDD_COLUMNS,
        }
