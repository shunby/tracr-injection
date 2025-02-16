from datasets import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()
dataset_name = "HuggingFaceFW/fineweb-edu"
sample_name = "sample-10BT"

DATASETS_DIR = os.path.expanduser(os.getenv('DATASETS_DIR'))

print("Loading FineWeb sample-10BT...")
dataset = load_dataset(
    dataset_name,
    sample_name,
    split='train',
    streaming=True,
    cache_dir=DATASETS_DIR
)

for i, sample in enumerate(dataset):
    print(sample)
    if i == 4:
        break

