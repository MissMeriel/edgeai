# save as: enable_easy_tagging.py

"""
Configure FiftyOne for easy review status changes using tags

This script adds FiftyOne tags (not Classification fields) for easier UI interaction
"""

import fiftyone as fo
from fiftyone import ViewField as F
import argparse

def setup_easy_review_tagging(dataset_name: str):
    """
    Setup FiftyOne tags for easier review status management
    
    FiftyOne has two types of tagging:
    1. Classification fields (what we've been using) - harder to change in UI
    2. Sample tags - easy to add/remove with keyboard shortcuts
    """
    
    dataset = fo.load_dataset(dataset_name)
    
    print(f"\n{'='*70}")
    print("SETTING UP EASY REVIEW TAGGING")
    print(f"{'='*70}")
    
    # Convert tag_review_status to simple tags
    print("\nConverting review status to FiftyOne tags...")
    
    for sample in dataset.iter_samples(progress=True):
        # Get current review status
        if hasattr(sample, 'tag_review_status') and sample.tag_review_status:
            status = sample.tag_review_status.label
            
            # Add as simple tag
            if status not in sample.tags:
                sample.tags.append(status)
                sample.save()
    
    print("\n✅ Setup complete!")
    print("\nNow in FiftyOne UI:")
    print("  1. Select sample(s)")
    print("  2. Press 't' to open tag menu")
    print("  3. Type 'fixed' and press Enter")
    print("  4. The sample is now tagged as 'fixed'!")
    
    # Create tag-based views
    dataset.save_view("tagged_todo", dataset.match_tags("todo"), overwrite=True)
    dataset.save_view("tagged_fixed", dataset.match_tags("fixed"), overwrite=True)
    dataset.save_view("tagged_needs_work", dataset.match_tags("needs_work"), overwrite=True)
    
    print("\n📁 Created tag-based views:")
    print("  - tagged_todo")
    print("  - tagged_fixed")
    print("  - tagged_needs_work")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', help='Dataset name')
    args = parser.parse_args()
    
    setup_easy_review_tagging(args.dataset_name)