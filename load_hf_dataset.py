"""Download the published HMDA 2024 dataset from Hugging Face and export it
locally as CSV so it can be used as HMDA_DATA_PATH for the analysis script.

Usage:
    python3 load_hf_dataset.py
    export HMDA_DATA_PATH="$(python3 -c 'import load_hf_dataset; print(load_hf_dataset.DEFAULT_EXPORT_PATH)')"
    python3 hmda_loan_analysis_gender_geo_artifactno1.py
"""

import os
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "retrogradespace/hmda_2024"

# Defaults to a path outside the repo so a multi-GB export doesn't get synced
# through OneDrive/iCloud or accidentally picked up by git.
DEFAULT_EXPORT_PATH = os.path.expanduser("~/.cache/hmda_2024/hmda_2024_lar.csv")


def main():
    export_path = Path(os.environ.get("HMDA_EXPORT_PATH", DEFAULT_EXPORT_PATH))
    export_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset '{DATASET_ID}' from Hugging Face...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(dataset)
    print(dataset[0])

    print(f"Exporting to {export_path} ...")
    dataset.to_csv(export_path)
    print(f"Saved: {export_path}")
    print(
        "\nTo run the analysis against this file:\n"
        f'  export HMDA_DATA_PATH="{export_path}"\n'
        "  python3 hmda_loan_analysis_gender_geo_artifactno1.py"
    )


if __name__ == "__main__":
    main()
