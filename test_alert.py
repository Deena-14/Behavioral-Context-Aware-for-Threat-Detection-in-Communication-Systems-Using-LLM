from llm_analysis import LLMThreatAnalyzer
from threat_alert import ThreatAlertManager

# simple event representing a failed login
evt = {
    'source': 'auth',
    'user': 'badguy',
    'action': 'login',
    'status': 'failed',
    'details': 'invalid password attempt'
}

# run the analysis and create an alert
analysis = LLMThreatAnalyzer().analyze_threat_context(str(evt))
alert = ThreatAlertManager().create_alert(analysis)

print(analysis)
print(alert)
