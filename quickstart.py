"""
Quick Start Script for Multimodal LLM-Based Cybersecurity System
Run this script to quickly execute threat detection
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print system banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    🔒 MULTIMODAL LLM-BASED CYBERSECURITY SYSTEM - QUICK START                 ║
║                                                                               ║
║    Context-Aware Threat Detection for Communication Networks                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_directories():
    """Create required directories"""
    print("\n📁 Creating required directories...")
    directories = ['logs', 'results', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  {directory}/")
    print("✅ Directories created!")

def run_single_analysis():
    """Run single threat analysis"""
    print("\n" + "="*80)
    print("🔍 Running Single Threat Analysis...")
    print("="*80)
    subprocess.run([sys.executable, "main.py", "--mode", "single"])

def run_interactive_analysis():
    """Run interactive mode"""
    print("\n" + "="*80)
    print("🎯 Running Interactive Mode...")
    print("="*80)
    print("\nControls:")
    print("  - Press Enter: Run next analysis")
    print("  - 'q': Quit")
    print("  - 's': View statistics")
    print("\n" + "="*80)
    subprocess.run([sys.executable, "main.py", "--mode", "interactive"])

def run_batch_analysis(iterations=5):
    """Run batch analysis"""
    print("\n" + "="*80)
    print(f"📊 Running Batch Analysis ({iterations} iterations)...")
    print("="*80)
    subprocess.run([sys.executable, "main.py", "--mode", "batch", "--iterations", str(iterations)])

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("\n" + "="*80)
    print("📈 Launching Dashboard...")
    print("="*80)
    print("\n🌐 Dashboard will open at: http://localhost:8501")
    print("   (If not, manually open the URL in your browser)")
    print("\n📋 Features in Dashboard:")
    print("  - Real-time threat visualization")
    print("  - Active alert management")
    print("  - Network traffic analysis")
    print("  - Threat statistics")
    print("  - System configuration")
    print("  - Alert export")
    print("\n" + "="*80)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard closed")

def display_menu():
    """Display main menu"""
    menu = """
╔════════════════════════════════════════════════════════════════╗
║                     EXECUTION MENU                             ║
╚════════════════════════════════════════════════════════════════╝

Choose execution mode:

1️⃣  Single Analysis
    → Run one complete threat detection cycle
    → Quick results in ~30 seconds
    → Best for: Testing, single threat check

2️⃣  Interactive Mode
    → Continuous analysis with user control
    → Run multiple analyses with real-time interaction
    → Best for: Monitoring, learning, exploration

3️⃣  Batch Analysis
    → Multiple iterations for comprehensive testing
    → Aggregated statistics and reports
    → Best for: Performance testing, multiple scenarios

4️⃣  Launch Dashboard
    → Web-based visualization interface
    → Real-time threat monitoring
    → Alert management and configuration
    → Best for: Visualization, management, alerts

5️⃣  View Logs
    → Display system logs
    → Useful for debugging and monitoring

6️⃣  Exit
    → Close the application

────────────────────────────────────────────────────────────────
    """
    print(menu)

def view_logs():
    """View system logs"""
    log_file = Path("logs/system.log")
    if log_file.exists():
        print("\n" + "="*80)
        print("📋 SYSTEM LOGS")
        print("="*80 + "\n")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Show last 50 lines
            for line in lines[-50:]:
                print(line.rstrip())
    else:
        print("⚠️  No logs found. Run an analysis first.")

def main():
    """Main menu system"""
    print_banner()
    
    # Create directories
    create_directories()
    
    # Main loop
    while True:
        display_menu()
        
        try:
            choice = input("\n👉 Select option (1-6): ").strip()
            
            if choice == '1':
                run_single_analysis()
            elif choice == '2':
                run_interactive_analysis()
            elif choice == '3':
                try:
                    iterations = input("📊 Number of iterations (default: 5): ").strip()
                    iterations = int(iterations) if iterations else 5
                    run_batch_analysis(iterations)
                except ValueError:
                    print("❌ Invalid number. Using default (5).")
                    run_batch_analysis(5)
            elif choice == '4':
                launch_dashboard()
            elif choice == '5':
                view_logs()
            elif choice == '6':
                print("\n👋 Thank you for using the Multimodal Cybersecurity System!")
                print("✅ Goodbye!\n")
                break
            else:
                print("❌ Invalid option. Please select 1-6.")
                continue
            
            # Ask to continue
            print("\n" + "="*80)
            cont = input("\n🔄 Return to menu? (Press Enter or 'y' for yes, 'n' for no): ").strip().lower()
            if cont == 'n':
                print("\n👋 Thank you for using the Multimodal Cybersecurity System!")
                print("✅ Goodbye!\n")
                break
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("👋 Thank you for using the Multimodal Cybersecurity System!")
            print("✅ Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
