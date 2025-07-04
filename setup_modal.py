#!/usr/bin/env python3
"""
Setup script for deploying the QA system to Modal.

This script helps you set up Modal and deploy your QA system.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_modal_installation():
    """Check if Modal is installed."""
    try:
        import modal
        print(f"✅ Modal is installed (version: {modal.__version__})")
        return True
    except ImportError:
        print("❌ Modal is not installed")
        return False

def main():
    print("🚀 Modal QA System Setup")
    print("=" * 50)
    
    # Check if Modal is installed
    if not check_modal_installation():
        print("\n📦 Installing Modal...")
        if not run_command("pip install modal", "Installing Modal"):
            print("Failed to install Modal. Please install it manually: pip install modal")
            return
    
    # Check if user is authenticated with Modal
    print("\n🔐 Checking Modal authentication...")
    result = subprocess.run("modal token current", shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Not authenticated with Modal")
        print("\n🔑 Setting up Modal authentication...")
        print("Please follow the instructions to authenticate with Modal:")
        if not run_command("modal setup", "Setting up Modal authentication"):
            print("Failed to set up Modal authentication. Please run 'modal setup' manually.")
            return
    else:
        print("✅ Already authenticated with Modal")
    
    # Deploy the application
    print("\n🚀 Deploying QA System to Modal...")
    deploy_success = run_command("modal deploy modal_deploy.py", "Deploying QA System")
    
    if deploy_success:
        print("\n🎉 Deployment successful!")
        print("\n📋 Next steps:")
        print("1. Your QA system is now live on Modal")
        print("2. Check your Modal dashboard for the deployment URL")
        print("3. For development with hot-reloading, use: modal serve modal_deploy.py")
        print("4. To check logs: modal logs qa-system")
        print("5. To stop the deployment: modal app stop qa-system")
        
        # Try to get the app URL
        print("\n🔍 Getting app information...")
        run_command("modal app list", "Listing Modal apps")
    else:
        print("\n❌ Deployment failed")
        print("Please check the error messages above and try again.")
        print("\nFor development/testing, you can use: modal serve modal_deploy.py")

if __name__ == "__main__":
    main()
