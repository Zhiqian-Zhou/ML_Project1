"""Backward-compatible training entrypoint delegating directly to run.py --train."""

import sys
import run

if __name__ == "__main__":
    run.main(["--train"] + sys.argv[1:])
