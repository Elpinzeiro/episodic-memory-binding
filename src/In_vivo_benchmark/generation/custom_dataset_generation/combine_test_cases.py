#!/usr/bin/env python3
"""
Script to combine all JSON test cases from test_cases_claude and test_cases_gpt folders
into a single unified JSON file.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def combine_test_cases(folder1: str, folder2: str, output_file: str):
    """
    Combine all JSON files from two folders into a single file.
    
    Args:
        folder1: Path to first folder (e.g., 'test_cases_claude')
        folder2: Path to second folder (e.g., 'test_cases_gpt')
        output_file: Path to output combined JSON file
    """
    combined_test_cases = {}
    
    # Process both folders
    for folder in [folder1, folder2]:
        folder_path = Path(folder)
        
        if not folder_path.exists():
            print(f"Warning: Folder '{folder}' does not exist. Skipping...")
            continue
        
        # Find all JSON files in the folder
        json_files = list(folder_path.glob("*.json"))
        print(f"\nProcessing {len(json_files)} JSON files from '{folder}'...")
        
        for json_file in json_files:
            print(f"  Reading: {json_file.name}")
            data = load_json_file(json_file)
            
            # Extract test_cases from the file
            if "test_cases" in data:
                test_cases = data["test_cases"]
                
                # Add each test case to combined dictionary
                for test_id, test_data in test_cases.items():
                    if test_id in combined_test_cases:
                        print(f"    Warning: Duplicate test case ID '{test_id}' found. Skipping duplicate.")
                    else:
                        combined_test_cases[test_id] = test_data
                        print(f"    Added test case: {test_id}")
            else:
                print(f"    Warning: No 'test_cases' field found in {json_file.name}")
    
    # Create final combined structure
    combined_data = {
        "test_cases": combined_test_cases
    }
    
    # Write to output file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Successfully combined {len(combined_test_cases)} test cases")
    print(f"✓ Output written to: {output_file}")
    
    return combined_data


def main():
    """Main function."""
    # Define folder paths
    folder1 = "test_cases_claude"
    folder2 = "test_cases_gpt"
    output_file = "combined_test_cases_new.json"
    
    # Check if running from uploaded data directory
    upload_folder1 = os.path.join("generated_prompts", "test_cases_claude")
    upload_folder2 = os.path.join("generated_prompts", "test_cases_gpt5")
    
    # Try uploaded folders first
    if os.path.exists(upload_folder1) or os.path.exists(upload_folder2):
        print("Using folders in generated_prompts/")
        folder1 = upload_folder1
        folder2 = upload_folder2
    else:
        print("Looking for folders in current directory...")
    
    # Combine test cases
    combined_data = combine_test_cases(folder1, folder2, output_file)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total test cases: {len(combined_data['test_cases'])}")
    print(f"  Test case IDs:")
    for test_id in combined_data['test_cases'].keys():
        print(f"    - {test_id}")


if __name__ == "__main__":
    main()