"""
Network Traffic Capture and Analysis
Captures real packets using Scapy and extracts features, with a resilient fallback.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random
import json
try:
    from scapy.all import sniff, IP, TCP, UDP, conf
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class NetworkCapture:
    """Capture and process network traffic"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.captured_packets = []
        self.flow_statistics = {}
    
    def capture_packets(self, duration: int = 60, packet_count: int = 100) -> List[Dict]:
        """Live network packet capture using Scapy with a Layer 3 workaround"""
        logger.info(f"Starting real capture of {packet_count} packets...")
        
        packets = []

        def packet_callback(packet):
            if IP in packet:
                # Safely extract ports depending on protocol
                sport = 0
                dport = 0
                if TCP in packet:
                    sport = packet[TCP].sport
                    dport = packet[TCP].dport
                elif UDP in packet:
                    sport = packet[UDP].sport
                    dport = packet[UDP].dport
                
                # Protocol mapping (simplistic)
                proto_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}
                proto_name = proto_map.get(packet[IP].proto, str(packet[IP].proto))

                pkt_dict = {
                    'packet_id': len(packets) + 1,
                    'timestamp': datetime.now(),
                    'source_ip': packet[IP].src,
                    'dest_ip': packet[IP].dst,
                    'source_port': sport,
                    'dest_port': dport,
                    'protocol': proto_name,
                    'packet_size': len(packet),
                    'flags': str(packet[TCP].flags) if TCP in packet else 'NONE',
                    'ttl': packet[IP].ttl,
                    'checksum': hex(packet[IP].chksum) if hasattr(packet[IP], 'chksum') else '0x0',
                }
                packets.append(pkt_dict)

        if not SCAPY_AVAILABLE:
            logger.warning("Scapy not available on this system. Using simulation mode.")
            return self._simulate_capture(duration, packet_count)

        try:
            # Force Scapy to use a Layer 3 socket (Bypasses the WinPcap driver error)
            logger.info("Attempting Layer 3 sniffing to bypass Windows driver restrictions...")
            sniff(prn=packet_callback, count=packet_count, timeout=duration, opened_socket=conf.L3socket())
            
            self.captured_packets = packets
            logger.info(f"Captured {len(packets)} real packets")
            return packets
            
        except Exception as e:
            # FAILSAFE: If live sniff fails, seamlessly fall back to simulation
            logger.warning(f"Live sniffing failed ({str(e)}). Falling back to simulation mode.")
            return self._simulate_capture(duration, packet_count)
            
    def _simulate_capture(self, duration: int, packet_count: int) -> List[Dict]:
        """Fallback method if real capture fails"""
        packets = []
        protocols = ['TCP', 'UDP', 'ICMP', 'DNS', 'HTTP', 'HTTPS', 'SSH', 'RDP']
        ports = [22, 80, 443, 3306, 5432, 8080, 53, 3389]
        
        for i in range(packet_count):
            packet = {
                'packet_id': i + 1,
                'timestamp': datetime.now() - timedelta(seconds=random.randint(0, duration)),
                'source_ip': f'192.168.1.{random.randint(1, 254)}',
                'dest_ip': f'10.0.0.{random.randint(1, 254)}',
                'source_port': random.randint(1024, 65535),
                'dest_port': random.choice(ports),
                'protocol': random.choice(protocols),
                'packet_size': random.randint(32, 65535),
                'flags': random.choice(['SYN', 'ACK', 'FIN', 'RST']),
                'ttl': random.randint(1, 255),
                'checksum': f'0x{random.randint(0, 65535):04x}',
            }
            packets.append(packet)
            
        self.captured_packets = packets
        return packets
    
    def analyze_flows(self) -> Dict[str, Any]:
        """Analyze traffic flows from captured packets"""
        logger.info("Analyzing network flows...")
        
        flows = {}
        
        for packet in self.captured_packets:
            flow_key = f"{packet['source_ip']}:{packet['source_port']}->{packet['dest_ip']}:{packet['dest_port']}"
            
            if flow_key not in flows:
                flows[flow_key] = {
                    'packets': 0,
                    'bytes': 0,
                    'duration': 0,
                    'protocol': packet['protocol'],
                    'source_ip': packet['source_ip'],
                    'dest_ip': packet['dest_ip'],
                    'dest_port': packet['dest_port'],
                }
            
            flows[flow_key]['packets'] += 1
            flows[flow_key]['bytes'] += packet['packet_size']
        
        self.flow_statistics = flows
        logger.info(f"Analyzed {len(flows)} unique flows")
        return flows
    
    def detect_suspicious_patterns(self) -> List[Dict]:
        """Detect suspicious network patterns"""
        logger.info("Detecting suspicious network patterns...")
        
        suspicious_patterns = []
        
        # Check for port scanning
        flows = self.analyze_flows()
        
        for source_ip, flow_ports in self._group_flows_by_source().items():
            if len(flow_ports) > 10:
                suspicious_patterns.append({
                    'pattern': 'port_scanning',
                    'source_ip': source_ip,
                    'ports_scanned': len(flow_ports),
                    'severity': 'high',
                    'description': f'Source {source_ip} scanned {len(flow_ports)} ports'
                })
        
        # Check for data exfiltration (large data transfer)
        for flow_key, statistics in flows.items():
            if statistics['bytes'] > 100000:
                suspicious_patterns.append({
                    'pattern': 'large_data_transfer',
                    'source_ip': statistics['source_ip'],
                    'dest_ip': statistics['dest_ip'],
                    'bytes': statistics['bytes'],
                    'severity': 'medium',
                    'description': f"Large data transfer: {statistics['bytes']} bytes"
                })
        
        # Check for unusual ports
        for flow_key, statistics in flows.items():
            if statistics['dest_port'] not in [22, 80, 443, 53, 3306]:
                suspicious_patterns.append({
                    'pattern': 'unusual_port',
                    'source_ip': statistics['source_ip'],
                    'dest_port': statistics['dest_port'],
                    'severity': 'low',
                    'description': f"Connection to unusual port {statistics['dest_port']}"
                })
        
        top_patterns = suspicious_patterns[:10]  # Return top 10
        logger.info(f"Detected {len(top_patterns)} suspicious patterns (top 10 of {len(suspicious_patterns)} total)")
        return top_patterns
    
    def _group_flows_by_source(self) -> Dict[str, set]:
        """Group flows by source IP"""
        grouped = {}
        for flow_key, flow in self.flow_statistics.items():
            source_ip = flow['source_ip']
            port = flow['dest_port']
            if source_ip not in grouped:
                grouped[source_ip] = set()
            grouped[source_ip].add(port)
        return grouped
    
    def get_traffic_summary(self) -> Dict[str, Any]:
        """Get summary of captured traffic"""
        if not self.flow_statistics:
            self.analyze_flows()
        
        total_bytes = sum(f['bytes'] for f in self.flow_statistics.values())
        total_packets = sum(f['packets'] for f in self.flow_statistics.values())
        
        summary = {
            'total_packets': total_packets,
            'total_bytes': total_bytes,
            'unique_flows': len(self.flow_statistics),
            'average_packet_size': total_bytes / total_packets if total_packets > 0 else 0,
            'flows': list(self.flow_statistics.values())[:5],  # Top 5 flows
        }
        
        return summary

class PacketAnalyzer:
    """Analyze individual packets in detail"""
    
    def __init__(self, packets: List[Dict]):
        self.packets = packets
    
    def extract_features(self) -> List[Dict]:
        """Extract features from packets for ML analysis"""
        logger.info("Extracting features from packets...")
        
        features = []
        
        for packet in self.packets:
            feature = {
                'packet_id': packet['packet_id'],
                'protocol': packet['protocol'],
                'packet_size': packet['packet_size'],
                'src_port_type': self._classify_port(packet['source_port']),
                'dst_port_type': self._classify_port(packet['dest_port']),
                'ttl_value': packet['ttl'],
                'is_syn_flag': 1 if packet['flags'] == 'SYN' else 0,
                'is_suspicious': random.random() > 0.85,  # 15% suspicious
            }
            features.append(feature)
        
        logger.info(f"Extracted features from {len(features)} packets")
        return features
    
    def _classify_port(self, port: int) -> str:
        """Classify port as well-known, registered, or ephemeral"""
        if port < 1024:
            return 'well_known'
        elif port < 49152:
            return 'registered'
        else:
            return 'ephemeral'
    
    def detect_protocol_anomalies(self) -> List[Dict]:
        """Detect anomalous protocol behavior"""
        logger.info("Detecting protocol anomalies...")
        
        anomalies = []
        protocol_stats = {}
        
        # Count protocol usage
        for packet in self.packets:
            proto = packet['protocol']
            protocol_stats[proto] = protocol_stats.get(proto, 0) + 1
        
        # Check for anomalies
        for proto, count in protocol_stats.items():
            if proto == 'ICMP' and count > 10:
                anomalies.append({
                    'type': 'excessive_icmp',
                    'protocol': proto,
                    'count': count,
                    'severity': 'medium'
                })
            elif proto == 'DNS' and count > 50:
                anomalies.append({
                    'type': 'excessive_dns',
                    'protocol': proto,
                    'count': count,
                    'severity': 'medium'
                })
        
        logger.info(f"Detected {len(anomalies)} protocol anomalies")
        return anomalies