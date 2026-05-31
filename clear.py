import os
import shutil
import sys
from pathlib import Path

# Force UTF-8 encoding for printing to handle Vietnamese characters in terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def clear_directory_contents(directory_path):
    """
    Deletes all files and subdirectories inside the given directory,
    but keeps the directory itself.
    """
    path = Path(directory_path)
    
    if not path.exists():
        print(f"Directory not found, skipping: {path}")
        return

    print(f"Clearing contents of: {path}")
    
    # Iterate over all items in the directory
    for item in path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
                print(f"  Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"  Deleted directory: {item.name}")
        except Exception as e:
            print(f"  Failed to delete {item}: {e}")

def main():
    # Base directory of the project
    base_dir = Path(__file__).parent
    
    # List of directories to clear (relative to base_dir)
    target_dirs = [
        "data/audio",
        "data/aligned",
        "data/annotations/auto",
        "data/annotations/compare",
        "data/annotations/wav2vec2_raw",
        "data/annotations/wavlm_features",
        "data/annotations/wavlm_raw",
        "data/final",
        "data/meta",
        "data/processed/frames",
        "data/processed/mouth",
        "data/processed/clips"
    ]
    
    print("Starting data cleanup...")
    print("-" * 30)
    
    for dir_rel_path in target_dirs:
        abs_path = base_dir / dir_rel_path
        clear_directory_contents(abs_path)
        
    print("-" * 30)
    print("Cleanup complete!")

if __name__ == "__main__":
    main()
