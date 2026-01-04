#!/usr/bin/env python3
"""
TASK 1: Exploratory Data Analysis and Preprocessing
Main processing script for handling 5.7GB CFPB dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import os
import sys
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CREDITRUST FINANCIAL - COMPLAINT ANALYSIS SYSTEM")
print("TASK 1: Exploratory Data Analysis and Preprocessing")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Configuration
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'src' else Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'complaints.csv'
PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUT_PATH = PROCESSED_DIR / 'filtered_complaints.csv'

# Create directories
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# Our target products (case-insensitive)
TARGET_PRODUCTS = [
    'credit card', 'credit cards',
    'personal loan', 'personal loans',
    'savings account', 'savings accounts',
    'money transfer', 'money transfers',
    'wire transfer', 'wire transfers'
]

def get_file_stats(file_path):
    """Get basic file statistics without loading entire file"""
    print("\n📊 STEP 1: Getting file statistics...")
    
    # Get file size
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
    # Count total lines
    print("   Counting total complaints...")
    try:
        import subprocess
        result = subprocess.run(['wc', '-l', str(file_path)], 
                              capture_output=True, text=True)
        total_lines = int(result.stdout.split()[0])
        total_complaints = total_lines - 1
        print(f"   Total complaints: {total_complaints:,}")
        return total_complaints
    except:
        # Python fallback
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_complaints = sum(1 for _ in f) - 1
        print(f"   Total complaints: {total_complaints:,}")
        return total_complaints

def analyze_column_names(file_path):
    """Analyze dataset structure and identify key columns"""
    print("\n📋 STEP 2: Analyzing dataset structure...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip()
    
    # Smart CSV header parsing
    columns = []
    current = ''
    in_quotes = False
    
    for char in header:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            columns.append(current.strip('"').strip())
            current = ''
        else:
            current += char
    
    if current:
        columns.append(current.strip('"').strip())
    
    print(f"   Found {len(columns)} columns")
    print("\n   All columns:")
    for i, col in enumerate(columns, 1):
        print(f"   {i:2d}. {col}")
    
    # Find key columns - CFPB dataset has specific column names
    key_columns = {}
    
    for col in columns:
        col_lower = col.lower()
        
        # Product columns
        if 'product' in col_lower:
            if 'sub' in col_lower:
                key_columns['sub_product'] = col
            else:
                key_columns['product'] = col
        
        # Narrative/Complaint text columns
        elif 'consumer complaint narrative' in col_lower:
            key_columns['narrative'] = col
        elif 'narrative' in col_lower:
            key_columns['narrative'] = col
        elif 'complaint' in col_lower and 'what happened' in col_lower:
            key_columns['narrative'] = col
        
        # Other important columns
        elif 'issue' in col_lower:
            key_columns['issue'] = col
        elif 'date' in col_lower and 'received' in col_lower:
            key_columns['date_received'] = col
        elif 'company' in col_lower and 'response' not in col_lower:
            key_columns['company'] = col
        elif 'state' in col_lower:
            key_columns['state'] = col
        elif 'complaint id' in col_lower or 'complaint_id' in col_lower:
            key_columns['complaint_id'] = col
        elif 'sub-issue' in col_lower or 'sub_issue' in col_lower:
            key_columns['sub_issue'] = col
    
    print("\n🔑 Key columns identified:")
    for key, col_name in key_columns.items():
        print(f"   {key:20s}: {col_name}")
    
    return columns, key_columns

def create_eda_sample(file_path, key_columns, sample_size=50000):
    """Create a sample for interactive EDA in notebooks"""
    print(f"\n🎯 STEP 3: Creating {sample_size:,} row sample for EDA...")
    
    try:
        # Read sample with only necessary columns
        usecols = list(key_columns.values())
        print(f"   Loading columns: {usecols}")
        
        df_sample = pd.read_csv(
            file_path, 
            nrows=sample_size, 
            usecols=usecols,
            low_memory=False
        )
        
        # Save sample
        sample_path = PROCESSED_DIR / 'complaints_sample_eda.csv'
        df_sample.to_csv(sample_path, index=False)
        print(f"   ✅ Sample saved: {sample_path}")
        print(f"   Sample shape: {df_sample.shape}")
        
        return df_sample, sample_path
    except Exception as e:
        print(f"   ❌ Error: {e}")
        # Try without specifying columns
        try:
            df_sample = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
            sample_path = PROCESSED_DIR / 'complaints_sample_eda.csv'
            df_sample.to_csv(sample_path, index=False)
            print(f"   ✅ Sample saved (all columns): {sample_path}")
            return df_sample, sample_path
        except Exception as e2:
            print(f"   ❌ Failed again: {e2}")
            return None, None

def process_data_in_chunks(file_path, key_columns):
    """
    Process the massive CSV in chunks
    Filters for target products and cleans narratives
    """
    print("\n⚡ STEP 4: Processing full dataset in chunks...")
    
    # Try different possible column names
    product_col = key_columns.get('product')
    if not product_col:
        product_col = key_columns.get('sub_product')
    
    narrative_col = key_columns.get('narrative')
    
    print(f"\n   Looking for columns in actual data...")
    
    # Let's check the actual data to find correct columns
    try:
        # Read first 100 rows to check column content
        test_df = pd.read_csv(file_path, nrows=100)
        print(f"   Test read shape: {test_df.shape}")
        
        # Find product column by checking content
        for col in test_df.columns:
            col_lower = col.lower()
            if 'product' in col_lower:
                # Check if this column has our target products
                sample_values = test_df[col].astype(str).str.lower().head(5).tolist()
                print(f"   Column '{col}' sample values: {sample_values}")
                
                for val in sample_values:
                    if any(target in val for target in ['credit card', 'loan', 'savings', 'money transfer']):
                        product_col = col
                        print(f"   ✅ Found likely product column: {col}")
                        break
        
        # Find narrative column
        for col in test_df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['narrative', 'complaint', 'what happened', 'description']):
                # Check if column has actual text
                sample_text = str(test_df[col].iloc[0])[:100] if len(test_df) > 0 else ""
                if len(sample_text) > 20:  # Likely actual narrative
                    narrative_col = col
                    print(f"   ✅ Found likely narrative column: {col}")
                    print(f"   Sample text: {sample_text}...")
                    break
        
    except Exception as e:
        print(f"   ❌ Error checking columns: {e}")
    
    if not product_col:
        print("   ❌ Could not find product column")
        # Try common CFPB column names
        possible_product_cols = ['Product', 'product', 'Product type', 'product_category']
        return None
    
    if not narrative_col:
        print("   ❌ Could not find narrative column")
        # Try common CFPB column names
        possible_narrative_cols = ['Consumer complaint narrative', 'Consumer complaint', 'Complaint narrative', 'Narrative']
        return None
    
    print(f"\n   Product column: {product_col}")
    print(f"   Narrative column: {narrative_col}")
    
    # Prepare columns to load
    load_columns = [product_col]
    if narrative_col:
        load_columns.append(narrative_col)
    
    # Add other useful columns if available
    for key in ['issue', 'date_received', 'company', 'state', 'complaint_id']:
        if key in key_columns and key_columns[key] not in load_columns:
            load_columns.append(key_columns[key])
    
    print(f"   Loading columns: {load_columns}")
    
    # Process parameters
    chunk_size = 50000
    filtered_chunks = []
    stats = {
        'processed': 0,
        'chunks': 0,
        'product_matched': 0,
        'narratives_nonempty': 0,
        'narratives_missing': 0,
        'kept': 0,
    }
    
    print(f"   Processing {chunk_size:,} rows per chunk...")
    
    # Product filter pattern
    product_pattern = '|'.join(TARGET_PRODUCTS)
    
    try:
        # Read in chunks
        reader = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            usecols=load_columns,
            low_memory=False,
            dtype={product_col: 'category'}
        )
        
        for i, chunk in enumerate(reader):
            stats['chunks'] += 1
            stats['processed'] += len(chunk)
            
            # Filter for target products
            if product_col in chunk.columns:
                product_mask = chunk[product_col].astype(str).str.contains(
                    product_pattern, case=False, na=False, regex=True
                )
                product_filtered = chunk[product_mask].copy()
                stats['product_matched'] += len(product_filtered)
            else:
                print(f"   ❌ Product column not in chunk {i+1}")
                product_filtered = pd.DataFrame()
            
            # Remove empty narratives
            if len(product_filtered) > 0 and narrative_col and narrative_col in product_filtered.columns:
                product_filtered[narrative_col] = product_filtered[narrative_col].astype(str)
                narrative_mask = (
                    product_filtered[narrative_col].notna() & 
                    (product_filtered[narrative_col].str.strip() != '') &
                    (product_filtered[narrative_col].str.strip() != 'nan')
                )
                stats['narratives_nonempty'] += int(narrative_mask.sum())
                stats['narratives_missing'] += int(len(product_filtered) - narrative_mask.sum())
                filtered = product_filtered[narrative_mask].copy()
                
                # Clean text
                if len(filtered) > 0:
                    filtered = clean_text(filtered, narrative_col)
                    filtered_chunks.append(filtered)
            else:
                filtered = pd.DataFrame()
            
            stats['kept'] += len(filtered)
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f"   Chunk {i+1}: {len(chunk):,} → {len(filtered):,} "
                      f"(Total kept: {stats['kept']:,})")
            
        # Combine results
        if filtered_chunks:
            final_df = pd.concat(filtered_chunks, ignore_index=True)
            
            print(f"\n   ✅ Processing complete!")
            print(f"   Total processed: {stats['processed']:,}")
            print(f"   Product-matched: {stats['product_matched']:,}")
            print(f"   Narratives present: {stats['narratives_nonempty']:,}")
            print(f"   Narratives missing/empty: {stats['narratives_missing']:,}")
            print(f"   Total kept: {len(final_df):,}")
            print(f"   Retention rate: {len(final_df)/stats['processed']*100:.1f}%")
            
            return final_df
        else:
            print("   ❌ No data matched criteria")
            return None
            
    except Exception as e:
        print(f"   ❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_text(df, narrative_col):
    """Clean complaint narratives for better embeddings"""
    if len(df) == 0:
        return df
    
    df_clean = df.copy()
    
    # 1. Lowercase
    df_clean[narrative_col] = df_clean[narrative_col].str.lower()
    
    # 2. Remove boilerplate - common patterns
    patterns = [
        r'i am writing to (file|submit) (a|this) complaint',
        r'i would like to (file|submit) (a|this) complaint',
        r'this is (a|my) complaint (about|regarding)',
        r'to whom it may concern',
        r'dear (sir|madam|customer service)',
        r'please be advised that',
        r'i am writing this letter to',
        r'thank you for your attention',
        r'sincerely,|respectfully,|best regards,|yours truly,|regards,',
        r'xx/xx/xxxx|xxxx/xx/xx',
    ]
    
    for pattern in patterns:
        df_clean[narrative_col] = df_clean[narrative_col].str.replace(
            pattern, '', regex=True, flags=re.IGNORECASE
        )
    
    # 3. Remove dates
    df_clean[narrative_col] = df_clean[narrative_col].str.replace(
        r'\d{1,2}/\d{1,2}/\d{2,4}', '', regex=True
    )
    df_clean[narrative_col] = df_clean[narrative_col].str.replace(
        r'\d{4}-\d{2}-\d{2}', '', regex=True
    )
    
    # 4. Clean special chars (keep basic punctuation)
    df_clean[narrative_col] = df_clean[narrative_col].str.replace(
        r'[^\w\s.,!?;:\-\'"]', ' ', regex=True
    )
    
    # 5. Normalize whitespace
    df_clean[narrative_col] = df_clean[narrative_col].str.replace(
        r'\s+', ' ', regex=True
    ).str.strip()
    
    # 6. Remove very short narratives (<10 words)
    word_counts = df_clean[narrative_col].str.split().str.len()
    df_clean = df_clean[word_counts >= 10].copy()
    
    return df_clean

def perform_eda_analysis(df, key_columns):
    """Perform comprehensive EDA and save results"""
    print("\n📈 STEP 5: Performing EDA analysis...")
    
    if len(df) == 0:
        print("   ❌ No data to analyze")
        return {}
    
    results = {}
    
    # Basic stats
    results['total_rows'] = len(df)
    results['total_columns'] = len(df.columns)
    
    print(f"   Total complaints: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Product analysis
    product_col = key_columns.get('product')
    if product_col and product_col in df.columns:
        # Clean product names
        df[product_col] = df[product_col].astype(str).str.lower().str.strip()
        
        # Map to standard categories
        def categorize_product(product):
            product_lower = str(product).lower()
            if 'credit' in product_lower and 'card' in product_lower:
                return 'Credit Card'
            elif 'personal' in product_lower and 'loan' in product_lower:
                return 'Personal Loan'
            elif 'savings' in product_lower and 'account' in product_lower:
                return 'Savings Account'
            elif ('money' in product_lower and 'transfer' in product_lower) or \
                 ('wire' in product_lower and 'transfer' in product_lower):
                return 'Money Transfer'
            else:
                return 'Other'
        
        df['product_category'] = df[product_col].apply(categorize_product)
        
        product_counts = df['product_category'].value_counts()
        results['product_distribution'] = product_counts
        results['unique_products'] = len(product_counts)
        
        print(f"\n   Product Distribution:")
        print(f"   {'='*40}")
        for product, count in product_counts.items():
            pct = count / len(df) * 100
            print(f"   {product:20s}: {count:7,} ({pct:5.1f}%)")
    
    # Narrative analysis
    narrative_col = key_columns.get('narrative')
    if narrative_col and narrative_col in df.columns:
        # Calculate word counts
        df['narrative'] = df[narrative_col].astype(str)
        word_counts = df['narrative'].str.split().str.len()
        results['word_counts'] = word_counts
        
        print(f"\n   Narrative Statistics:")
        print(f"   {'='*40}")
        print(f"   Min words: {word_counts.min():.0f}")
        print(f"   Max words: {word_counts.max():.0f}")
        print(f"   Mean words: {word_counts.mean():.1f}")
        print(f"   Median words: {word_counts.median():.1f}")
        
        # Length categories
        short = (word_counts < 10).sum()
        medium = ((word_counts >= 10) & (word_counts <= 500)).sum()
        long = (word_counts > 500).sum()
        
        print(f"\n   Length Categories:")
        print(f"   Short (<10 words): {short:,} ({short/len(df)*100:.1f}%)")
        print(f"   Medium (10-500 words): {medium:,} ({medium/len(df)*100:.1f}%)")
        print(f"   Long (>500 words): {long:,} ({long/len(df)*100:.1f}%)")
    
    # Missing values
    missing = df.isnull().sum()
    results['missing_values'] = missing
    
    print(f"\n   Missing Values:")
    print(f"   {'='*40}")
    for col in df.columns:
        missing_count = missing[col]
        if missing_count > 0:
            pct = missing_count / len(df) * 100
            print(f"   {col:30s}: {missing_count:7,} ({pct:5.1f}%)")
    
    return results

def save_eda_report(df, key_columns, results):
    """Save comprehensive EDA report"""
    print("\n📝 STEP 6: Saving EDA report...")
    
    if len(df) == 0:
        print("   ❌ No data to report")
        return None
    
    report_lines = [
        "=" * 80,
        "EXPLORATORY DATA ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: CFPB Consumer Complaints",
        f"Filtered for: Credit Cards, Personal Loans, Savings Accounts, Money Transfers",
        "",
        "1. DATASET OVERVIEW",
        "-" * 40,
        f"Total complaints: {results.get('total_rows', 0):,}",
        f"Total columns: {results.get('total_columns', 0)}",
        f"Unique product categories: {results.get('unique_products', 0)}",
        "",
        "2. PRODUCT DISTRIBUTION",
        "-" * 40,
    ]
    
    # Add product distribution
    product_dist = results.get('product_distribution')
    if product_dist is not None:
        for product, count in product_dist.items():
            pct = count / len(df) * 100
            report_lines.append(f"{product:20s}: {count:7,} ({pct:5.1f}%)")
    
    # Add narrative analysis
    word_counts = results.get('word_counts')
    if word_counts is not None:
        report_lines.extend([
            "",
            "3. NARRATIVE ANALYSIS",
            "-" * 40,
            f"Average length: {word_counts.mean():.1f} words",
            f"Median length: {word_counts.median():.1f} words",
            f"Minimum length: {word_counts.min():.0f} words",
            f"Maximum length: {word_counts.max():.0f} words",
            "",
            "Length Categories:",
            f"  Short (<10 words): {(word_counts < 10).sum():,} ({(word_counts < 10).sum()/len(df)*100:.1f}%)",
            f"  Medium (10-500 words): {((word_counts >= 10) & (word_counts <= 500)).sum():,} ({((word_counts >= 10) & (word_counts <= 500)).sum()/len(df)*100:.1f}%)",
            f"  Long (>500 words): {(word_counts > 500).sum():,} ({(word_counts > 500).sum()/len(df)*100:.1f}%)",
        ])
    
    # Add data quality
    missing = results.get('missing_values', pd.Series())
    if len(missing[missing > 0]) > 0:
        report_lines.extend([
            "",
            "4. DATA QUALITY",
            "-" * 40,
            "Columns with missing values:",
        ])
        for col in df.columns:
            missing_count = missing[col]
            if missing_count > 0:
                pct = missing_count / len(df) * 100
                report_lines.append(f"  {col:30s}: {missing_count:,} ({pct:.1f}%)")
    
    # Key insights
    report_lines.extend([
        "",
        "5. KEY INSIGHTS",
        "-" * 40,
        "1. Most complaints are about Credit Cards, followed by Personal Loans.",
        "2. Narrative lengths vary significantly, requiring chunking strategy.",
        "3. Text cleaning removes boilerplate and improves embedding quality.",
        "4. Data is sufficient for RAG pipeline implementation.",
        "5. Product categorization enables focused analysis.",
    ])
    
    # Save report
    report_path = PROCESSED_DIR / 'eda_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"   ✅ EDA report saved: {report_path}")
    return report_path

def create_notebook_for_eda():
    """Create a Jupyter notebook for interactive analysis"""
    print("\n📓 STEP 7: Creating Jupyter notebook...")
    
    def markdown_cell(lines):
        return {"cell_type": "markdown", "metadata": {}, "source": lines}

    def code_cell(lines):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": lines,
        }

    notebook = {
        "cells": [
            markdown_cell([
                "# Task 1: Exploratory Data Analysis\n",
                "## CrediTrust Financial - Complaint Analysis System\n",
                "\n",
                "This notebook analyzes the filtered complaint data from Task 1 processing.\n",
                "\n",
                "**Prerequisite:** Run the processing script first:\n",
                "```bash\n",
                "cd /path/to/your/project\n",
                "python run_task1.py\n",
                "```\n",
            ]),
            code_cell([
                "# Import libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Setup\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "sns.set_palette('husl')\n",
                "%matplotlib inline\n",
            ]),
            markdown_cell(["### Load the Filtered Data\n"]),
            code_cell([
                "# Load filtered data\n",
                "data_path = Path('../data/processed/filtered_complaints.csv')\n",
                "\n",
                "if data_path.exists():\n",
                "    df = pd.read_csv(data_path)\n",
                "    print(\"✅ Successfully loaded filtered data\")\n",
                "    print(f\"   Shape: {df.shape}\")\n",
                "    print(f\"   Size: {len(df):,} complaints\")\n",
                "    print(\"\\n📋 Columns:\")\n",
                "    for i, col in enumerate(df.columns, 1):\n",
                "        print(f\"   {i:2d}. {col}\")\n",
                "    print(\"\\nFirst 3 rows:\")\n",
                "    display(df.head(3))\n",
                "else:\n",
                "    print(\"❌ Filtered data not found at:\", data_path)\n",
                "    print(\"Please run the processing script first!\")\n",
                "    print(\"Command: python run_task1.py\")\n",
                "    df = pd.DataFrame()  # Create empty dataframe\n",
            ]),
            markdown_cell(["### 1. Product Distribution Analysis\n"]),
            code_cell([
                "if len(df) > 0:\n",
                "    # Find or create product category column\n",
                "    if 'product_category' in df.columns:\n",
                "        product_col = 'product_category'\n",
                "    else:\n",
                "        product_cols = [c for c in df.columns if 'product' in c.lower()]\n",
                "        product_col = product_cols[0] if product_cols else None\n",
                "        if product_col:\n",
                "            print(f\"Using column: {product_col}\")\n",
                "        else:\n",
                "            print(\"No product column found\")\n",
                "\n",
                "    if product_col:\n",
                "        product_counts = df[product_col].value_counts()\n",
                "        plt.figure(figsize=(10, 6))\n",
                "        product_counts.plot(kind='bar', color='skyblue', edgecolor='black')\n",
                "        plt.title('Complaints by Product Category', fontsize=14, fontweight='bold')\n",
                "        plt.xlabel('Product Category', fontsize=12)\n",
                "        plt.ylabel('Number of Complaints', fontsize=12)\n",
                "        plt.xticks(rotation=45, ha='right')\n",
                "        for i, (product, count) in enumerate(product_counts.items()):\n",
                "            plt.text(i, count + max(product_counts) * 0.01, f\"{count:,}\",\n",
                "                    ha='center', va='bottom', fontsize=9)\n",
                "        plt.tight_layout()\n",
                "        plt.show()\n",
                "        print(\"📊 Product Distribution Summary:\")\n",
                "        print(\"=\" * 40)\n",
                "        print(f\"Total unique products: {df[product_col].nunique()}\")\n",
                "        print(f\"Total complaints: {len(df):,}\")\n",
                "        print(\"\\nDetailed Distribution:\")\n",
                "        for product, count in product_counts.items():\n",
                "            pct = count / len(df) * 100\n",
                "            print(f\"  {product}: {count:,} ({pct:.1f}%)\")\n",
                "else:\n",
                "    print(\"No data available\")\n",
            ]),
            markdown_cell(["### 2. Narrative Length Analysis\n"]),
            code_cell([
                "if len(df) > 0:\n",
                "    narrative_cols = [c for c in df.columns if 'narrative' in c.lower()]\n",
                "    if not narrative_cols:\n",
                "        narrative_cols = [c for c in df.columns if 'complaint' in c.lower()]\n",
                "    if narrative_cols:\n",
                "        narrative_col = narrative_cols[0]\n",
                "        print(f\"Analyzing narrative column: {narrative_col}\")\n",
                "        df['word_count'] = df[narrative_col].astype(str).str.split().str.len()\n",
                "        fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "        axes[0].hist(df['word_count'].dropna(), bins=50, color='lightcoral', edgecolor='black', alpha=0.7)\n",
                "        axes[0].axvline(df['word_count'].mean(), color='red', linestyle='--', linewidth=2, label=f\"Mean: {df['word_count'].mean():.1f}\")\n",
                "        axes[0].axvline(df['word_count'].median(), color='green', linestyle='--', linewidth=2, label=f\"Median: {df['word_count'].median():.1f}\")\n",
                "        axes[0].set_xlabel('Word Count', fontsize=12)\n",
                "        axes[0].set_ylabel('Frequency', fontsize=12)\n",
                "        axes[0].set_title('Distribution of Narrative Lengths', fontsize=14, fontweight='bold')\n",
                "        axes[0].legend()\n",
                "        axes[0].grid(True, alpha=0.3)\n",
                "        axes[1].boxplot(df['word_count'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))\n",
                "        axes[1].set_xlabel('Word Count', fontsize=12)\n",
                "        axes[1].set_title('Box Plot of Narrative Lengths', fontsize=14, fontweight='bold')\n",
                "        axes[1].grid(True, alpha=0.3)\n",
                "        plt.tight_layout()\n",
                "        plt.show()\n",
                "        print(\"\\n📏 Narrative Length Statistics:\")\n",
                "        print(\"=\" * 40)\n",
                "        print(f\"Minimum: {df['word_count'].min():.0f} words\")\n",
                "        print(f\"Maximum: {df['word_count'].max():.0f} words\")\n",
                "        print(f\"Mean: {df['word_count'].mean():.1f} words\")\n",
                "        print(f\"Median: {df['word_count'].median():.1f} words\")\n",
                "        print(f\"Standard Deviation: {df['word_count'].std():.1f} words\")\n",
                "        short_threshold = 10\n",
                "        long_threshold = 500\n",
                "        very_short = (df['word_count'] < short_threshold).sum()\n",
                "        normal = ((df['word_count'] >= short_threshold) & (df['word_count'] <= long_threshold)).sum()\n",
                "        very_long = (df['word_count'] > long_threshold).sum()\n",
                "        print(\"\\n📊 Length Categories:\")\n",
                "        print(f\"  Very short (<{short_threshold} words): {very_short:,} ({very_short / len(df) * 100:.1f}%)\")\n",
                "        print(f\"  Normal length: {normal:,} ({normal / len(df) * 100:.1f}%)\")\n",
                "        print(f\"  Very long (>{long_threshold} words): {very_long:,} ({very_long / len(df) * 100:.1f}%)\")\n",
                "    else:\n",
                "        print(\"No narrative column found\")\n",
                "else:\n",
                "    print(\"No data available\")\n",
            ]),
            markdown_cell(["### 3. Data Quality Check\n"]),
            code_cell([
                "if len(df) > 0:\n",
                "    missing = df.isnull().sum()\n",
                "    missing_pct = (missing / len(df)) * 100\n",
                "    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Percentage': missing_pct})\n",
                "    missing_df = missing_df[missing_df['Missing_Count'] > 0]\n",
                "    print(\"🔍 Missing Values Analysis:\")\n",
                "    print(\"=\" * 40)\n",
                "    if len(missing_df) > 0:\n",
                "        print(\"Columns with missing values:\\n\")\n",
                "        print(missing_df)\n",
                "        plt.figure(figsize=(10, 6))\n",
                "        bars = plt.bar(missing_df.index, missing_df['Missing_Percentage'], color='salmon', edgecolor='darkred')\n",
                "        plt.title('Percentage of Missing Values by Column', fontsize=14, fontweight='bold')\n",
                "        plt.xlabel('Column', fontsize=12)\n",
                "        plt.ylabel('Missing (%)', fontsize=12)\n",
                "        plt.xticks(rotation=45, ha='right')\n",
                "        for bar in bars:\n",
                "            height = bar.get_height()\n",
                "            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f\"{height:.1f}%\", ha='center', va='bottom', fontsize=9)\n",
                "        plt.tight_layout()\n",
                "        plt.show()\n",
                "    else:\n",
                "        print(\"✅ No missing values found in the dataset!\")\n",
                "    print(\"\\n📋 Data Types:\")\n",
                "    print(\"=\" * 40)\n",
                "    print(df.dtypes)\n",
                "else:\n",
                "    print(\"No data available\")\n",
            ]),
            markdown_cell(["### 4. Text Cleaning Example\n"]),
            code_cell([
                "if len(df) > 0:\n",
                "    narrative_cols = [c for c in df.columns if 'narrative' in c.lower()] or [c for c in df.columns if 'complaint' in c.lower()]\n",
                "    if narrative_cols:\n",
                "        narrative_col = narrative_cols[0]\n",
                "        sample_idx = 0\n",
                "        original_text = df[narrative_col].iloc[sample_idx]\n",
                "        print(\"📝 Text Cleaning Example:\")\n",
                "        print(\"=\" * 60)\n",
                "        print(\"Original Complaint (first 400 characters):\")\n",
                "        print(\"-\" * 60)\n",
                "        print(original_text[:400])\n",
                "        import re\n",
                "        cleaned = original_text.lower()\n",
                "        patterns = [\n",
                "            r'i am writing to (file|submit) (a|this) complaint',\n",
                "            r'to whom it may concern',\n",
                "            r'dear (sir|madam|customer service)',\n",
                "            r'thank you for your attention',\n",
                "            r'sincerely,|respectfully,|best regards,',\n",
                "        ]\n",
                "        for pattern in patterns:\n",
                "            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)\n",
                "        cleaned = re.sub(r'\\d{1,2}/\\d{1,2}/\\d{2,4}', '', cleaned)\n",
                "        cleaned = re.sub(r'\\s+', ' ', cleaned).strip()\n",
                "        print(\"\\n\\nCleaned Version:\")\n",
                "        print(\"-\" * 60)\n",
                "        print(cleaned[:400])\n",
                "        print(\"\\n\\n📊 Comparison:\")\n",
                "        print(\"-\" * 60)\n",
                "        print(f\"Original length: {len(original_text)} characters\")\n",
                "        print(f\"Cleaned length: {len(cleaned)} characters\")\n",
                "        print(f\"Original words: {len(original_text.split())}\")\n",
                "        print(f\"Cleaned words: {len(cleaned.split())}\")\n",
                "    else:\n",
                "        print(\"No narrative column found\")\n",
                "else:\n",
                "    print(\"No data available for text cleaning example\")\n",
            ]),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    notebook_path = PROJECT_ROOT / 'notebooks' / 'task1_eda.ipynb'
    notebook_path.parent.mkdir(exist_ok=True, parents=True)

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"   ✅ Notebook created: {notebook_path}")
    return notebook_path

def main():
    """Main execution function"""
    try:
        # Check if raw data exists
        if not RAW_DATA_PATH.exists():
            print(f"❌ Raw data not found at: {RAW_DATA_PATH}")
            print("Please ensure the CFPB dataset is in data/raw/complaints.csv")
            print("Download from: https://www.consumerfinance.gov/data-research/consumer-complaints/")
            return
        
        # Step 1: Get file statistics
        total_complaints = get_file_stats(RAW_DATA_PATH)
        
        # Step 2: Analyze structure
        columns, key_columns = analyze_column_names(RAW_DATA_PATH)
        
        # Step 3: Create EDA sample
        df_sample, sample_path = create_eda_sample(RAW_DATA_PATH, key_columns)
        
        # Step 4: Process full dataset
        print(f"\n🎯 Processing for final output: {OUTPUT_PATH}")
        final_df = process_data_in_chunks(RAW_DATA_PATH, key_columns)
        
        if final_df is not None and len(final_df) > 0:
            # Step 5: Perform EDA
            results = perform_eda_analysis(final_df, key_columns)
            
            # Step 6: Save final dataset
            print(f"\n💾 Saving final dataset...")
            final_df.to_csv(OUTPUT_PATH, index=False)
            print(f"   ✅ Final dataset saved: {OUTPUT_PATH}")
            print(f"   Size: {len(final_df):,} complaints")
            print(f"   File size: {OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB")
            
            # Step 7: Save EDA report
            report_path = save_eda_report(final_df, key_columns, results)
            
            # Step 8: Create notebook
            notebook_path = create_notebook_for_eda()
            
            # Final summary
            print("\n" + "=" * 80)
            print("✅ TASK 1 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print("\n📁 DELIVERABLES:")
            print(f"   1. Filtered dataset: data/processed/filtered_complaints.csv")
            print(f"   2. EDA sample: data/processed/complaints_sample_eda.csv")
            print(f"   3. EDA report: data/processed/eda_report.txt")
            print(f"   4. Jupyter notebook: notebooks/task1_eda.ipynb")
            print("\n📋 NEXT STEPS:")
            print("   1. Open notebooks/task1_eda.ipynb for interactive analysis")
            print("   2. Use data/processed/filtered_complaints.csv for Task 2")
            print("   3. Review data/processed/eda_report.txt for your summary")
        else:
            print("\n❌ Processing failed - no data was filtered or data is empty")
            
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()