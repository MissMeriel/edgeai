# save as: switch_to_simple_tags.py

"""
Convert Classification-based review status to simple tags
Much easier to use in the UI!
"""

import fiftyone as fo
from fiftyone import ViewField as F
import sys

def convert_to_simple_tags(dataset_name: str):
    """Convert tag_review_status Classification to simple sample tags"""
    
    dataset = fo.load_dataset(dataset_name)
    
    print(f"Converting review status to simple tags...")
    
    for sample in dataset.iter_samples(progress=True):
        # Get current review status
        if hasattr(sample, 'tag_review_status') and sample.tag_review_status:
            status = sample.tag_review_status.label
            
            # Remove any existing review status tags
            for old_status in ['todo', 'fixed', 'needs_work', 'reviewed', 'skip']:
                if old_status in sample.tags:
                    sample.tags.remove(old_status)
            
            # Add current status as simple tag
            sample.tags.append(status)
            sample.save()
    
    print(f"\n✅ Converted to simple tags!")
    print(f"\nNow in FiftyOne:")
    print(f"  1. Select sample(s)")
    print(f"  2. Press 't' key")
    print(f"  3. Type 'fixed' and press Enter")
    print(f"  4. Done! Much easier!")
    
    # Create tag-based views
    dataset.save_view("TODO", dataset.match_tags("todo"), overwrite=True)
    dataset.save_view("FIXED", dataset.match_tags("fixed"), overwrite=True)
    dataset.save_view("NEEDS_WORK", dataset.match_tags("needs_work"), overwrite=True)
    
    print(f"\n📁 Created simple views:")
    print(f"  - TODO ({len(dataset.match_tags('todo'))} samples)")
    print(f"  - FIXED ({len(dataset.match_tags('fixed'))} samples)")
    print(f"  - NEEDS_WORK ({len(dataset.match_tags('needs_work'))} samples)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python switch_to_simple_tags.py <dataset_name>")
        sys.exit(1)
    
    convert_to_simple_tags(sys.argv[1])