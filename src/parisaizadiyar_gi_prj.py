"""Compatibility entry point; use python -m gan_vae for new runs."""

from pathlib import Path
import sys


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gan_vae.__main__ import main

    main()
