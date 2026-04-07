from threat_alert import ThreatAlertManager

mgr = ThreatAlertManager()
print("Current alerts:")
print(mgr.alerts)
print(f"\nTotal alerts: {len(mgr.alerts)}")
