#!/usr/bin/env python3
"""
Runner script for Task 1
Execute from project root directory
"""

import sys
from pathlib import Path
import os

print("=" * 60)
print("CREDITRUST FINANCIAL - Task 1 Runner")
print("EDA and Preprocessing for Complaint Analysis")
print("=" * 60)

# Check current directory
current_dir = Path.cwd()
print(f"Current directory: {current_dir}")

# Check project structure
print("\n📁 Checking project structure...")
required_folders = ['src', 'data/raw', 'data/processed', 'notebooks']
all_exist = True

for folder in required_folders:
    folder_path = current_dir / folder
    if folder_path.exists():
        print(f"✅ {folder}/")
    else:
        print(f"❌ {folder}/ - Creating...")
        folder_path.mkdir(parents=True, exist_ok=True)
        all_exist = False

# Check for raw data
raw_data_path = current_dir / 'data' / 'raw' / 'complaints.csv'
if raw_data_path.exists():
    print(f"✅ Raw data: data/raw/complaints.csv")
else:
    print(f"❌ Raw data not found: data/raw/complaints.csv")
    print("\n📥 Please download the CFPB dataset:")
    print("   1. Go to: https://www.consumerfinance.gov/data-research/consumer-complaints/")
    print("   2. Download the full complaint dataset")
    print("   3. Save as 'complaints.csv' in data/raw/")
    print("\n   Or use direct link (may need updating):")
    print("   https://files.consumerfinance.gov/ccdb/complaints.csv.zip")
    sys.exit(1)

# Add src to path and run main script
print("\n🚀 Starting Task 1 processing...")
print("=" * 60)

# Add src to Python path
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

try:
    from task1_processing import main
    main()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure:")
    print("  1. You're in the project root directory")
    print("  2. src/task1_processing.py exists")
    print("\nProject structure should be:")
    print("  rag-complaint-chatbot/")
    print("  ├── src/")
    print("  │   └── task1_processing.py  <-- This file")
    print("  ├── data/")
    print("  │   └── raw/complaints.csv   <-- Your data here")
    print("  └── run_task1.py             <-- Run this")
except Exception as e:
    print(f"❌ Execution error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Task 1 runner completed")
print("Check the outputs in data/processed/ and notebooks/")
print("=" * 60)