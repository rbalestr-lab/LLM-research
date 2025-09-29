#!/usr/bin/env python3
"""
Test script for GoodBadInjection modifier.
This script demonstrates how the GoodBadInjection works with sample data.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spurious_corr.modifiers import GoodBadInjection
from spurious_corr.transform import spurious_transform, multi_label_spurious_transform
from datasets import Dataset

def test_goodbad_injection():
    """Test the GoodBadInjection modifier with sample data."""
    
    # Create sample dataset
    sample_data = {
        "text": [
            "This movie is fantastic and well-acted.",
            "The film was terrible and boring.",
            "Amazing cinematography and great story.",
            "Poor direction and bad acting.",
            "Excellent performance by the lead actor.",
            "Worst movie I've ever seen."
        ],
        "labels": [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    }
    
    dataset = Dataset.from_dict(sample_data)
    
    print("Original dataset:")
    for i, example in enumerate(dataset):
        print(f"  {i}: Label {example['labels']} - {example['text']}")
    
    # Test different injection configurations
    test_configs = [
        {"location": "beginning", "token_proportion": 0.1, "name": "Beginning, 10%"},
        {"location": "end", "token_proportion": 0.1, "name": "End, 10%"},
        {"location": "random", "token_proportion": 0.1, "name": "Random, 10%"},
        {"location": "beginning", "token_proportion": 0.2, "name": "Beginning, 20%"},
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        
        # Create modifier
        modifier = GoodBadInjection(
            location=config["location"],
            token_proportion=config["token_proportion"],
            seed=42
        )
        
        # Apply transformation to all examples (100% proportion)
        modified_dataset = spurious_transform(
            label_to_modify=1,  # Modify positive examples
            dataset=dataset,
            modifier=modifier,
            text_proportion=1.0,  # Modify all examples with label 1
            seed=42
        )
        
        print("Modified dataset (label 1 examples):")
        for i, example in enumerate(modified_dataset):
            if example['labels'] == 1:  # Only show positive examples
                print(f"  {i}: Label {example['labels']} - {example['text']}")
    
    # Test with both labels using the new multi-label transform
    print(f"\n--- Testing both labels with multi_label_spurious_transform (beginning, 10%) ---")
    modifier = GoodBadInjection(location="beginning", token_proportion=0.1, seed=42)
    
    # Use the new multi-label transform to modify both labels at once
    dataset_modified_both = multi_label_spurious_transform(
        labels_to_modify=[0, 1],  # Modify both label 0 and label 1
        dataset=dataset,
        modifier=modifier,
        text_proportion=1.0,  # Modify all examples
        seed=42
    )
    
    print("Modified dataset (both labels using multi_label_spurious_transform):")
    for i, example in enumerate(dataset_modified_both):
        print(f"  {i}: Label {example['labels']} - {example['text']}")
    
    # Compare with the old approach (two separate calls)
    print(f"\n--- Comparing with old approach (two separate calls) ---")
    modifier_old = GoodBadInjection(location="beginning", token_proportion=0.1, seed=42)
    
    # First modify label 1 examples
    dataset_modified_1 = spurious_transform(
        label_to_modify=1,
        dataset=dataset,
        modifier=modifier_old,
        text_proportion=1.0,
        seed=42
    )
    
    # Then modify label 0 examples
    dataset_modified_both_old = spurious_transform(
        label_to_modify=0,
        dataset=dataset_modified_1,
        modifier=modifier_old,
        text_proportion=1.0,
        seed=42
    )
    
    print("Modified dataset (both labels using old approach):")
    for i, example in enumerate(dataset_modified_both_old):
        print(f"  {i}: Label {example['labels']} - {example['text']}")

if __name__ == "__main__":
    test_goodbad_injection()
