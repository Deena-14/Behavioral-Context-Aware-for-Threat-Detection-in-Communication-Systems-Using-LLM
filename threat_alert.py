"""
Threat Alert Management
Generates, manages, and escalates security alerts
"""

import logging
import json
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
from config import SystemConfig

logger = logging.getLogger(__name__)

class ThreatAlertManager:
    """Manage threat alerts and escalation"""
    
    def __init__(self):
        self.alerts = []
        self.alert_history = []
        self.alert_cooldown = {}
        self.config = SystemConfig()
    
    def create_alert(self, threat_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create an alert from threat analysis results"""
        logger.info("Creating threat alert...")
        
        ta = threat_analysis.get('threat_analysis', {})
        risk_score = threat_analysis.get('overall_risk_score', 0.0)
        confidence = ta.get('confidence', 0.0)
        systems    = ta.get('affected_systems', [])
        alert = {
            'alert_id': f"ALERT-{len(self.alerts) + 1:06d}",
            'timestamp': datetime.now().isoformat(),
            'threat_type': ta.get('threat_type', 'unknown'),
            'risk_level': threat_analysis.get('overall_risk_level', 'low'),
            # canonical fields (used by llm_analysis / new dashboard)
            'overall_risk_score': risk_score,
            'severity_score': threat_analysis.get('severity_score', 0.0),
            'confidence': confidence,
            'affected_systems': systems,
            'recommended_actions': threat_analysis.get('recommendations', []),
            'mitre_technique': ta.get('knowledge_base_entry', {}).get('mitre_technique', 'N/A'),
            # legacy aliases (kept for backward-compat with old dashboard / threat_alert internals)
            'risk_score': risk_score,
            'confidence_score': confidence,
            'source_ips': systems,
            'description': ta.get('pattern', 'Unknown threat detected'),
            'actions_recommended': threat_analysis.get('recommendations', []),
            'status': 'active',
            'investigation_required': threat_analysis.get('overall_risk_level', 'low') in ['high', 'critical'],
        }
        
        self.alerts.append(alert)
        self.alert_history.append(alert)
        
        logger.info(f"Alert created: {alert['alert_id']} - {alert['threat_type']}")
        return alert
    
    def filter_alert_cooldown(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed due to cooldown"""
        # support either source_ips or affected_systems
        sources = alert.get('source_ips') or alert.get('affected_systems', [])
        alert_key = f"{alert['threat_type']}_{sources}"
        
        last_alert_time = self.alert_cooldown.get(alert_key)
        current_time = datetime.now()
        
        if last_alert_time:
            time_diff = (current_time - last_alert_time).total_seconds()
            if time_diff < SystemConfig.THREAT_CONFIG['alert_cooldown']:
                logger.info(f"Alert filtered due to cooldown: {alert_key}")
                return False
        
        self.alert_cooldown[alert_key] = current_time
        return True
    
    def generate_alert_summary(self) -> Dict[str, Any]:
        """Generate summary of current alerts"""
        logger.info("Generating alert summary...")
        
        summary = {
            'total_alerts': len(self.alerts),
            'active_alerts': len([a for a in self.alerts if a['status'] == 'active']),
            'critical_alerts': len([a for a in self.alerts if a['risk_level'] == 'critical']),
            'high_risk_alerts': len([a for a in self.alerts if a['risk_level'] == 'high']),
            'alerts_by_type': self._count_alerts_by_type(),
            'timeline': self._get_alert_timeline(),
        }
        
        return summary
    
    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by threat type"""
        counts = defaultdict(int)
        for alert in self.alerts:
            counts[alert['threat_type']] += 1
        return dict(counts)
    
    def _get_alert_timeline(self) -> List[Dict]:
        """Get alert timeline for visualization"""
        timeline = []
        for alert in sorted(self.alerts, key=lambda x: x['timestamp'])[:10]:
            timeline.append({
                'timestamp': alert['timestamp'],
                'threat_type': alert['threat_type'],
                'risk_level': alert['risk_level'],
            })
        return timeline
    
    def acknowledge_alert(self, alert_id: str, notes: str = "") -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged_time'] = datetime.now().isoformat()
                alert['acknowledgment_notes'] = notes
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolution: str = "") -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_time'] = datetime.now().isoformat()
                alert['resolution'] = resolution
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def escalate_alert(self, alert_id: str) -> bool:
        """Escalate an alert to higher severity"""
        for alert in self.alerts:
            if alert['alert_id'] == alert_id:
                risk_level_order = ['low', 'medium', 'high', 'critical']
                current_idx = risk_level_order.index(alert['risk_level'])
                if current_idx < len(risk_level_order) - 1:
                    alert['risk_level'] = risk_level_order[current_idx + 1]
                    alert['escalated'] = True
                    alert['escalation_time'] = datetime.now().isoformat()
                    logger.info(f"Alert escalated: {alert_id}")
                    return True
        return False

class AlertNotifier:
    """Send alerts through various channels"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or SystemConfig.ALERT_CONFIG
        self.notifications = []
    
    def send_email_alert(self, alert: Dict[str, Any], recipient: str) -> bool:
        """Send alert via email"""
        logger.info(f"Sending email alert to {recipient}...")
        
        email_content = self._format_alert_email(alert)
        
        # In production, use actual email service
        self.notifications.append({
            'type': 'email',
            'recipient': recipient,
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert['alert_id'],
            'content': email_content,
        })
        
        logger.info(f"Email alert sent to {recipient}")
        return True
    
    def send_slack_alert(self, alert: Dict[str, Any], webhook_url: str) -> bool:
        """Send alert to Slack"""
        logger.info("Sending Slack alert...")
        
        slack_message = self._format_slack_message(alert)
        
        # In production, use actual Slack API
        self.notifications.append({
            'type': 'slack',
            'webhook': webhook_url,
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert['alert_id'],
            'message': slack_message,
        })
        
        logger.info("Slack alert sent")
        return True
    
    def log_alert_file(self, alert: Dict[str, Any]) -> bool:
        """Log alert to file"""
        logger.info("Logging alert to file...")
        
        try:
            with open(self.config['log_file'], 'a') as f:
                f.write(json.dumps(alert, indent=2, default=str) + '\n')
                f.write('-' * 80 + '\n')
            logger.info(f"Alert logged to {self.config['log_file']}")
            return True
        except Exception as e:
            logger.error(f"Error logging alert: {str(e)}")
            return False
    
    def send_dashboard_notification(self, alert: Dict[str, Any]) -> bool:
        """Send notification to dashboard"""
        logger.info("Sending dashboard notification...")
        
        self.notifications.append({
            'type': 'dashboard',
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert['alert_id'],
            'alert': alert,
        })
        
        return True
    
    def broadcast_alert(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """Send alert through all configured channels"""
        logger.info(f"Broadcasting alert {alert['alert_id']}...")
        
        results = {
            'email': False,
            'slack': False,
            'log_file': False,
            'dashboard': False,
        }
        
        # Log to file (always enabled)
        results['log_file'] = self.log_alert_file(alert)
        
        # Send to dashboard
        results['dashboard'] = self.send_dashboard_notification(alert)
        
        # Send email if enabled
        if self.config['email_enabled']:
            results['email'] = self.send_email_alert(alert, 'security@company.com')
        
        # Send to Slack if enabled
        if self.config['slack_enabled']:
            results['slack'] = self.send_slack_alert(alert, 'webhook_url')
        
        logger.info(f"Alert broadcast completed: {results}")
        return results
    
    def _format_alert_email(self, alert: Dict[str, Any]) -> str:
        """Format alert for email"""
        return f"""
SECURITY ALERT NOTIFICATION

Alert ID: {alert['alert_id']}
Timestamp: {alert['timestamp']}
Threat Type: {alert['threat_type']}
Risk Level: {alert['risk_level'].upper()}
Risk Score: {alert['risk_score']}

Description: {alert['description']}

Affected Systems: {', '.join(alert['source_ips'])}

Recommended Actions:
{chr(10).join(f"  - {action}" for action in alert['actions_recommended'])}

Confidence Score: {alert['confidence_score']}

Please review and respond accordingly.
        """
    
    def _format_slack_message(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert for Slack"""
        
        risk_color_map = SystemConfig.RISK_LEVELS
        risk_color = risk_color_map.get(alert['risk_level'], {}).get('color', '#FF0000')
        
        return {
            'attachments': [
                {
                    'color': risk_color,
                    'title': f"⚠️ {alert['threat_type'].upper()} DETECTED",
                    'text': alert['description'],
                    'fields': [
                        {
                            'title': 'Risk Level',
                            'value': alert['risk_level'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Risk Score',
                            'value': str(alert['risk_score']),
                            'short': True
                        },
                        {
                            'title': 'Alert ID',
                            'value': alert['alert_id'],
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert['timestamp'],
                            'short': True
                        }
                    ]
                }
            ]
        }

class AlertAnalytics:
    """Analyze and report on alerts"""
    
    def __init__(self, alerts: List[Dict]):
        self.alerts = alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistical analysis of alerts"""
        
        stats = {
            'total_alerts': len(self.alerts),
            'average_risk_score': sum(a.get('risk_score', 0) for a in self.alerts) / len(self.alerts) if self.alerts else 0,
            'alerts_by_risk_level': self._count_by_risk_level(),
            'alerts_by_threat_type': self._count_by_threat_type(),
        }
        
        return stats
    
    def _count_by_risk_level(self) -> Dict[str, int]:
        """Count alerts by risk level"""
        counts = defaultdict(int)
        for alert in self.alerts:
            counts[alert.get('risk_level', 'unknown')] += 1
        return dict(counts)
    
    def _count_by_threat_type(self) -> Dict[str, int]:
        """Count alerts by threat type"""
        counts = defaultdict(int)
        for alert in self.alerts:
            counts[alert.get('threat_type', 'unknown')] += 1
        return dict(counts)
    
    def generate_report(self) -> str:
        """Generate alert analysis report"""
        stats = self.get_alert_statistics()
        
        report = f"""
THREAT ALERT ANALYSIS REPORT
Generated: {datetime.now().isoformat()}

SUMMARY:
  Total Alerts: {stats['total_alerts']}
  Average Risk Score: {stats['average_risk_score']:.2f}

ALERTS BY RISK LEVEL:
  {json.dumps(stats['alerts_by_risk_level'], indent=4)}

ALERTS BY THREAT TYPE:
  {json.dumps(stats['alerts_by_threat_type'], indent=4)}

        """
        return report
