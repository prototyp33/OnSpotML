import os
import sys

# Ensure repository root is on Python path so that 'src' and other top-level modules are importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Optionally also add the `src/` folder itself for direct module imports inside it (legacy)
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR) 