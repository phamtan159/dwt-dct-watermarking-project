"""
Compatibility entrypoint for the main baseline.

The current pipeline baseline learns from phoneme-level speech attributes,
visual attributes, and rule flags. The older visual-only CNN3D baseline is
available as train/train_visual_baseline.py.
"""

from train_attribute_classifier import main


if __name__ == "__main__":
    main()
