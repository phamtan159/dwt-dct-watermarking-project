import os
import shutil
import sys

# Ensure stdout can handle Vietnamese characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def clear_directory_contents(directory_path):
    """
    Deletes all files and subdirectories within the specified directory.
    The root directory itself is preserved.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return

    print(f"Clearing contents of: {directory_path}")
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"  Deleted file: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"  Deleted folder: {item}")
        except Exception as e:
            print(f"  Failed to delete {item_path}. Reason: {e}")

if __name__ == "__main__":
    # List of target folders to clear
    target_folders = [
        "data/aligned",
        "data/annotations/auto",
        "data/annotations/compare",
        "data/annotations/wav2vec2_raw",
        "data/annotations/wavlm_features",
        "data/annotations/wavlm_raw",
        "data/final",
        "data/meta",
        "data/processed/clips"
    ]

    # Get the absolute path of the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for folder in target_folders:
        folder_path = os.path.join(base_dir, folder)
        clear_directory_contents(folder_path)
    
    print("\nCleanup completed.")
