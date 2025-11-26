"""
Compatibility wrapper for environments that expect the legacy
`train_nsf_sim_cache_sid_load_pretretrain.py` entry point.

The original script name is `train_nsf_sim_cache_sid_load_pretrain.py`; this
module simply re-exports its `main` function so callers using the misspelled
path continue to work.
"""

from train_nsf_sim_cache_sid_load_pretrain import main


if __name__ == "__main__":
    main()
