# 🔒 Multimodal AI-Driven Cybersecurity System using LLMs

## Overview

A comprehensive threat detection system for modern communication networks that combines **multimodal data integration** with **Large Language Models (LLMs)** for context-aware, intelligent threat analysis.

### Key Features

✅ **Multimodal Data Integration**: Combines network traffic, system logs, IDS alerts, DNS queries, and endpoint activity  
✅ **LLM-Based Threat Detection**: Context-aware analysis using transformer models  
✅ **Retrieval-Augmented Reasoning**: Leverages threat intelligence knowledge base  
✅ **Real-Time Alert Management**: Automatic threat alert generation and escalation  
✅ **Interactive Dashboard**: Web-based visualization using Streamlit  
✅ **Comprehensive Reporting**: Detailed threat analysis and recommendations  

---

## Project Structure

```
project_1/
├── main.py                     # Main execution script
├── config.py                   # System configuration
├── multimodal_processor.py     # Data processing from multiple sources
├── llm_analysis.py             # LLM-based threat analysis
├── network_capture.py          # Network traffic capture & analysis
├── threat_alert.py             # Alert management system
├── streamlit_app.py            # Web dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── logs/                       # System logs
├── results/                    # Analysis results
└── data/                       # Data files
```

---

## Installation

### 1. Install Dependencies

```bash
cd c:\Users\Harshitha\OneDrive\Desktop\project_1
pip install -r requirements.txt
```

### 2. Create Required Directories

```bash
mkdir logs
mkdir results
mkdir data
```

---

## Execution Modes

The system supports three execution modes:

### **Single Mode** (Default)
Run one complete threat detection cycle:

```bash
python main.py --mode single
```

**Output:**
- Complete threat analysis results
- Alert generation
- JSON report

---

### **Interactive Mode**
Continuous analysis with user control:

```bash
python main.py --mode interactive
```

**Features:**
- Run multiple analyses
- View real-time statistics
- Interactive menu
- Options: Press Enter for next, 'q' to quit, 's' for stats

---

### **Batch Mode**
Multiple iterations for comprehensive testing:

```bash
# Run 5 iterations (default)
python main.py --mode batch

# Run custom number of iterations
python main.py --mode batch --iterations 10
```

**Output:**
- Batch execution summary
- Aggregated statistics
- JSON results file

---

## Dashboard Access

Launch the interactive web dashboard:

```bash
streamlit run streamlit_app.py
```

**Dashboard Features:**
- Real-time threat visualization
- Active alert management
- Network traffic analysis
- Threat statistics & analytics
- System configuration
- Alert export

Dashboard URL: `http://localhost:8501`

---

## System Architecture

### **Data Flow**

```
Network Traffic          System Logs           IDS Alerts
    ↓                       ↓                      ↓
Network Capture       Log Processing        Alert Processing
    ↓                       ↓                      ↓
    └───────────────── Multimodal Processor ─────────────┘
                            ↓
                    Data Aggregation & Correlation
                            ↓
                    LLM Threat Analysis
                            ↓
                Retrieval-Augmented Reasoning
                            ↓
                    Risk Assessment
                            ↓
                Alert Generation & Management
                            ↓
                    Dashboard & Reporting
```

---

## Threat Detection Workflow

### **Step 1: Network Traffic Capture**
- Captures packets from network interface
- Analyzes traffic flows
- Detects suspicious patterns (port scanning, data exfiltration)

### **Step 2: Multimodal Data Processing**
- Aggregates data from 5 sources:
  - Network logs
  - System logs
  - IDS alerts
  - DNS queries
  - Endpoint activity
- Correlates events across sources

### **Step 3: LLM-Based Threat Analysis**
- Context-aware analysis using language models
- Not rule-based but intelligence-driven
- Analyzes patterns and anomalies

### **Step 4: Retrieval-Augmented Reasoning**
- Searches threat intelligence knowledge base
- Matches indicators to known threat patterns
- Provides remediation recommendations

### **Step 5: Alert Generation**
- Creates structured threat alerts
- Applies cooldown to prevent duplicates
- Generates risk scores and confidence levels

### **Step 6: Notification**
Sends alerts through multiple channels:
- Email notifications
- Slack integration
- System logs
- Dashboard display

---

## Configuration

Edit `config.py` to customize:

```python
# LLM Configuration
LLM_CONFIG = {
    'provider': 'openai',
    'model_name': 'gpt-3.5-turbo',
    'temperature': 0.7,
    'max_tokens': 1024,
}

# Threat Detection Settings
THREAT_CONFIG = {
    'risk_threshold': 0.6,
    'alert_cooldown': 60,
}

# Data Sources
DATA_SOURCES = {
    'network_logs': True,
    'system_logs': True,
    'ids_alerts': True,
    'dns_logs': True,
    'endpoint_activity': True,
}
```

---

## Example Output

### Single Mode Results:

```
================================================================================
MULTIMODAL LLM-BASED CYBERSECURITY SYSTEM
Context-Aware Threat Detection for Communication Networks
================================================================================

[STEP 1] Network Traffic Capture
✓ Captured 100 packets, 45 flows
✓ Detected 8 suspicious patterns

[STEP 2] Multimodal Data Processing
✓ Aggregated 55 events from multiple sources
✓ Found 12 event correlations

[STEP 3] LLM-Based Threat Analysis
✓ Threat Analysis Completed
  - Threat Type: data_exfiltration
  - Risk Level: HIGH
  - Risk Score: 0.82

[STEP 4] Retrieval-Augmented Reasoning
✓ Knowledge Base Retrieval Completed
  - Matching Threats: 3
  - Remediation: Block outbound traffic, isolate system, investigate

[STEP 5] Alert Generation & Management
✓ Alert Generated: ALERT-000001
✓ Notifications Sent: {'log_file': True, 'dashboard': True, 'email': False, 'slack': False}

[STEP 6] Summary & Reporting
✓ Alert Summary Generated:
  - Total Alerts: 1
  - Critical Alerts: 0
  - High Risk Alerts: 1

================================================================================
PIPELINE COMPLETED SUCCESSFULLY
================================================================================
```

---

## API Components

### **MultimodalDataProcessor**
Aggregates and correlates data from multiple sources.

```python
processor = MultimodalDataProcessor(config)
data = processor.process_all()
# Returns: aggregated_data, correlations, llm_input
```

### **LLMThreatAnalyzer**
Performs context-aware threat analysis using LLMs.

```python
analyzer = LLMThreatAnalyzer(config)
results = analyzer.perform_threat_analysis(multimodal_data)
# Returns: threat_analysis, anomaly_analysis, correlation_analysis, recommendations
```

### **ThreatAlertManager**
Manages threat alerts and escalation.

```python
alert_manager = ThreatAlertManager()
alert = alert_manager.create_alert(analysis_result)
alert_manager.acknowledge_alert(alert_id)
alert_manager.escalate_alert(alert_id)
```

### **AlertNotifier**
Sends alerts through multiple channels.

```python
notifier = AlertNotifier(config)
notifier.broadcast_alert(alert)  # Email, Slack, log file, dashboard
```

### **NetworkCapture**
Captures and analyzes network traffic.

```python
capture = NetworkCapture(config)
packets = capture.capture_packets(packet_count=100)
flows = capture.analyze_flows()
suspicious = capture.detect_suspicious_patterns()
```

---

## Threat Categories

The system detects:

- 🔴 **Credential Stuffing** - Multiple login attempts with invalid credentials
- 🔴 **Brute Force** - Rapid authentication failure attempts
- 🔴 **Data Exfiltration** - Unauthorized data transfer to external networks
- 🔴 **DDoS Attack** - Distributed denial of service attacks
- 🔴 **Insider Threat** - Suspicious activity from authorized users
- 🔴 **APT Activity** - Advanced persistent threat indicators
- 🟠 **Malware** - Malicious software signatures
- 🟠 **Suspicious DNS** - Queries to suspicious domains
- 🟠 **Port Scanning** - Network reconnaissance activity
- 🟠 **Privilege Escalation** - Unauthorized privilege elevation

---

## Risk Levels

| Level | Score | Color | Action |
|-------|-------|-------|--------|
| 🔴 Critical | 0.9-1.0 | Red | Immediate action required |
| 🟠 High | 0.7-0.9 | Orange | Urgent investigation |
| 🟡 Medium | 0.4-0.7 | Yellow | Review and monitor |
| 🟢 Low | 0.0-0.4 | Green | Log for reference |

---

## Customization

### Add Custom Threat Category

Edit `config.py`:

```python
# Add to THREAT_CATEGORIES
'custom_threat': {
    'indicators': ['indicator1', 'indicator2'],
    'severity': 'high',
    'remediation': 'Your remediation steps'
}
```

### Integrate with Real LLM

Edit `llm_analysis.py`:

```python
import openai

def call_openai_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
```

### Add Email Notifications

Edit `config.py`:

```python
ALERT_CONFIG = {
    'email_enabled': True,
    'slack_enabled': True,
    # ... other config
}
```

---

## Logging

All system logs are saved to `logs/system.log`:

```bash
tail -f logs/system.log
```

Alert logs are saved to `logs/alerts.log`.

---

## Results & Reports

Analysis results are saved in `results/` directory:

- `batch_execution_*.json` - Batch mode results
- `alerts_*.json` - Alert exports
- Visualization plots

---

## Performance Metrics

The system provides comprehensive metrics:

- **Accuracy** - Correct threat identification rate
- **Precision** - True positive rate
- **Recall** - Threat detection rate
- **F1-Score** - Balanced performance measure
- **Confidence Scores** - LLM model confidence
- **Risk Scores** - Threat severity assessment

---

## Troubleshooting

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Port Already in Use (Dashboard)

```bash
streamlit run streamlit_app.py --server.port 8502
```

### Logs Directory Error

```bash
mkdir logs results data
```

---

## Advanced Features

### Retrieval-Augmented Reasoning

The system uses knowledge base retrieval to improve threat interpretation:

```python
rar_result = analyzer.retrieval_augmented_reasoning(
    threat_type='data_exfiltration',
    indicators=['large_transfer', 'unusual_ip', 'admin_access']
)
# Returns: matching_knowledge, enriched_analysis, remediation
```

### Multi-Source Correlation

Events from different sources are automatically correlated:
- Network data + IDS alerts
- System logs + Endpoint activity
- DNS logs + Network traffic

### Context-Aware Analysis

LLM understands context beyond simple pattern matching:
- Attack timeline reconstruction
- Attacker intent inference
- Lateral movement detection
- Data exfiltration paths

---

## Contributing

To extend the system:

1. Add new data sources in `multimodal_processor.py`
2. Define new threat categories in `config.py`
3. Update LLM prompts in `config.py`
4. Add visualizations in `streamlit_app.py`

---

## License

This project is for educational and research purposes.

---

## Support

For issues and questions:
1. Check system logs: `logs/system.log`
2. Review configuration: `config.py`
3. Enable debug logging in `main.py`

---

## References

### Academic Basis
- Multimodal Machine Learning
- Large Language Models for Security
- Threat Intelligence Integration
- Network Intrusion Detection

### Technologies Used
- Python 3.8+
- Transformers (Hugging Face)
- Streamlit
- Pandas, NumPy, Matplotlib
- OpenAI API

---

**Last Updated:** March 2026  
🔒 **Stay Secure!**
