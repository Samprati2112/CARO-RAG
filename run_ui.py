"""
Launcher script for CARO RAG vs Baseline RAG Comparative Evaluation Studio.
Usage:
    python run_ui.py
"""

import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def main():
    print("=" * 70)
    print(" Starting CARO RAG vs Baseline RAG Comparative Studio...")
    print("=" * 70)
    
    server_script = Path(__file__).resolve().parent / "server.py"
    if not server_script.exists():
        print(f"Error: Could not find {server_script}")
        sys.exit(1)
        
    print("Opening browser at http://localhost:8000 ...")
    time.sleep(1.0)
    webbrowser.open("http://localhost:8000")
    
    # Run server
    subprocess.run([sys.executable, str(server_script)])

if __name__ == "__main__":
    main()
