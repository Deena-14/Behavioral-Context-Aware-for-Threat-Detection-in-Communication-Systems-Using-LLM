"""
Simple filesystem watcher that converts access attempts into alerts
using the existing threat analysis pipeline.

Requires the ``watchdog`` package (pip install watchdog).
Run alongside the dashboard; it will add alerts in real time.
"""
import time
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from llm_analysis import LLMThreatAnalyzer
from threat_alert import ThreatAlertManager, AlertNotifier

analyzer = LLMThreatAnalyzer()
alert_mgr = ThreatAlertManager()
notifier = AlertNotifier()

class AccessEventHandler(FileSystemEventHandler):
    """Convert file system events into our alert schema."""

    def on_modified(self, event):
        # treat modifications as "access attempts" for demo purposes
        if event.is_directory:
            return
        evt = {
            'source': 'filesystem',
            'user': 'unknown',  # could use os.getlogin() or similar
            'action': 'modify',
            'resource': event.src_path,
            'status': 'attempt',
            'details': f"{event.event_type} {event.src_path}"
        }
        self.handle_event(evt)

    def on_created(self, event):
        if event.is_directory:
            return
        evt = {
            'source': 'filesystem',
            'user': 'unknown',
            'action': 'create',
            'resource': event.src_path,
            'status': 'attempt',
            'details': f"{event.event_type} {event.src_path}"
        }
        self.handle_event(evt)

    def on_deleted(self, event):
        if event.is_directory:
            return
        evt = {
            'source': 'filesystem',
            'user': 'unknown',
            'action': 'delete',
            'resource': event.src_path,
            'status': 'attempt',
            'details': f"{event.event_type} {event.src_path}"
        }
        self.handle_event(evt)

    def handle_event(self, evt: dict):
        analysis = analyzer.analyze_threat_context(str(evt))
        alert = alert_mgr.create_alert(analysis)
        if alert:
            # notify immediately; email config may need real credentials
            notifier.send_email_alert(alert, "secops@example.com")
            print(f"Generated alert: {alert['alert_id']} for {evt['resource']}")


def main(path_to_watch: str):
    event_handler = AccessEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()

    print(f"Watching {path_to_watch} for file events...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python filesystem_monitor.py <directory-to-watch>")
        sys.exit(1)
    main(sys.argv[1])
