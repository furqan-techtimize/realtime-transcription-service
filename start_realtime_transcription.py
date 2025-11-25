"""
Startup script for real-time transcription demo
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env file")
    pass

def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'websockets': 'websockets',
        'deepgram': 'deepgram-sdk',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is NOT installed")
    
    return missing

def install_dependencies(packages):
    """Install missing packages"""
    if not packages:
        return True
    
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    
    print("\n🔑 API Key Status:")
    if deepgram_key:
        print(f"✅ DEEPGRAM_API_KEY found: {deepgram_key[:8]}...")
    else:
        print("⚠️  DEEPGRAM_API_KEY not found in environment")
        print("   Set it with: $env:DEEPGRAM_API_KEY='your-api-key'")
    
    return deepgram_key is not None

def main():
    print("=" * 60)
    print("🎤 REAL-TIME TRANSCRIPTION SERVICE STARTUP")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("📋 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        response = input("Install missing packages? (y/n): ")
        if response.lower() == 'y':
            if not install_dependencies(missing):
                print("❌ Failed to install dependencies. Please install manually:")
                for package in missing:
                    print(f"   pip install {package}")
                return
        else:
            print("❌ Cannot start without required dependencies")
            return
    
    # Check API keys
    has_keys = check_api_keys()
    if not has_keys:
        print("\n⚠️  Warning: No API keys found. The service will not work without them.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Get paths
    service_script = Path(__file__).parent / 'realtime_transcription_service.py'
    demo_html = Path(__file__).parent / 'realtime_transcription_demo.html'
    
    if not service_script.exists():
        print(f"❌ Service script not found: {service_script}")
        return
    
    if not demo_html.exists():
        print(f"❌ Demo HTML not found: {demo_html}")
        return
    
    print("\n🚀 Starting service...")
    print(f"   Service: {service_script}")
    print(f"   Demo: {demo_html}")
    print()
    
    # Start the service
    try:
        process = subprocess.Popen(
            [sys.executable, str(service_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a moment for service to start
        time.sleep(2)
        
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("❌ Service failed to start:")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            return
        
        print(f"✅ Service started (PID: {process.pid})")
        print(f"🌐 WebSocket server: ws://localhost:8766")
        print()
        
        # Open browser
        print("🌐 Opening demo in browser...")
        webbrowser.open(f'file://{demo_html.absolute()}')
        
        print()
        print("=" * 60)
        print("✅ SERVICE IS RUNNING")
        print("=" * 60)
        print()
        print("📋 What to do:")
        print("   1. Click 'Start Recording' in the browser")
        print("   2. Grant microphone permissions")
        print("   3. Speak into your microphone")
        print("   4. Watch real-time transcription appear")
        print()
        print("🐛 Debug logs:")
        print("   - Watch the debug log section in the browser")
        print("   - Check this terminal for service logs")
        print()
        print("🛑 To stop: Press Ctrl+C")
        print("=" * 60)
        print()
        
        # Show service output
        try:
            while True:
                output = process.stderr.readline()
                if output:
                    print(output.strip())
                if process.poll() is not None:
                    break
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping service...")
            process.terminate()
            process.wait()
            print("✅ Service stopped")
        
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
