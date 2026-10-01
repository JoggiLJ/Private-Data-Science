import os

from datasets import get_dataset_config_names, load_dataset

DATASET_NAME = "edinburgh-dawg/mmlu-redux-2.0"

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MMLU Redux")

subjects = get_dataset_config_names(DATASET_NAME)
print(f"Found {len(subjects)} subjects: {subjects}")

for subject in subjects:
    subject_dir = os.path.join(output_dir, subject)
    if os.path.isdir(subject_dir):
        print(f"Skipping {subject}, already downloaded")
        continue

    print(f"Downloading {subject}...")
    ds = load_dataset(DATASET_NAME, subject)
    ds.save_to_disk(subject_dir)

print("Done.")