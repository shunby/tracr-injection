import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DATASETS_DIR = os.path.expanduser(os.getenv('DATASETS_DIR'))
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
SAMPLE_NAME = "sample-10BT"

def get_fineweb():
    dataset = load_dataset(
        DATASET_NAME,
        SAMPLE_NAME,
        split='train',
        streaming=True,
        cache_dir=DATASETS_DIR
    )

    return dataset
