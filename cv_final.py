"""Backward-compatibility forwarder for cross_validation.py.

All cross-validation and evaluation logic has been unified into cross_validation.py.
This script passes CLI arguments directly to cross_validation.main().
"""

import sys
from cross_validation import main

if __name__ == "__main__":
    main(sys.argv[1:])
