from dotenv import load_dotenv
import dill
import os

load_dotenv()

def load_model(directory=os.getenv('STORAGE_DIR'), filename=os.getenv('COMPILED_MODEL')):
    filepath = directory + '/' + filename
    filepath = os.path.expanduser(filepath)
    with open(filepath, 'rb') as f:
        return dill.load(f)


model = load_model()
model
bos='B'
model.apply([bos, '8', '8', '+', '3', '4', '=', '1', '2'])
