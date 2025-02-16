import csv
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def save_results(results: dict, prefix='experiments') -> None:
    """
    Save the results dictionary into a CSV file named 'prefix_{timestamp}.csv'.

    Args:
        results (dict): The dictionary to save.
        prefix (str): The prefix for the filename.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    storage_dir = os.getenv("STORAGE_DIR")
    experiments_dir = os.path.join(storage_dir, "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    file_path = os.path.join(experiments_dir, filename)

    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        for key, value in results.items():
            writer.writerow([key, value])

    print(f"Results saved to {file_path}")

