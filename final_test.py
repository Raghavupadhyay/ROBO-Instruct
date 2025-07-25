import numpy as np
from tensorflow.keras.models import load_model
import re
path="glove.6B.100d.txt"
# Load GloVe embeddings once
def load_glove(path=path, dim=100):
    embeddings = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            embeddings[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embeddings

embeddings_index = load_glove()
embedding_dim = 100

def get_task_embeddings(tasks):
    vecs = []
    for task in tasks:
        words = task.lower().split()
        we = [embeddings_index.get(w, np.zeros(embedding_dim)) for w in words]
        avg = np.mean(we, axis=0) if we else np.zeros(embedding_dim)
        vecs.append(avg)
    return np.array(vecs)

# Load model once
model = load_model("anuj_sir.h5")

mapping = [
    'forward', 'go_to_corridor', 'go_to_door', 'go_to_lift',
    'go_to_table', 'left', 'reverse', 'right',
    'roam_room', 'start', 'stop'
]
# Define your unit lists
list_angle = {'deg', 'degree', 'degrees', '°'}
list_speed = {'m/s', 'km/h', 'mph','m/sec','km/sec','kilometer','meter','kilometer/s','kilometer/second','kilometer/sec','meter/s','meter/second','meter/sec','m/second','km/second', 'mps', 'kph'}
list_distance = {'m', 'cm', 'mm', 'km', 'meter', 'meters', 'cm', 'mm','kilometer','kmeter'}

# Regex to match floats or ints followed by optional unit
num_pattern = re.compile(r'(?i)\b([+-]?\d+(?:\.\d*)?)\s*([a-zA-Z/°]+)?\b')

def parse_command(text):
    res = {'distance': None, 'angle': None, 'speed': None}
    for match in num_pattern.finditer(text):
        val = float(match.group(1))
        unit = (match.group(2) or '').lower()
        # Classify
        if unit in list_angle:
            key = 'angle'
        elif unit in list_speed:
            key = 'speed'
        elif unit in list_distance:
            key = 'distance'
        else:
            continue  # ignore unknown units or unit-less
        # Keep first valid match
        if res[key] is None:
            res[key] = (val, f"{val} {unit}".strip())
    return res

def infer_and_parse(prompt):
    parsed = parse_command(prompt)
    # Find which category is non‑None
    for typ, v in parsed.items():
        if v is not None:
            qty_type, qty_str = typ, v[1]
            break
    else:
        qty_type, qty_str = None, None

    # Run your model inference 
    emb = get_task_embeddings([prompt])
    preds = model.predict(emb)
    idx = np.argmax(preds)
    action = mapping[idx]

    # Print one‑line list
    print(f'["{action}", "{qty_type}", "{qty_str}"]')

if __name__ == "__main__":
    while True:
        inp = input("Enter the input: ")
        if inp.lower() in {'exit', 'quit'}:
            break
        infer_and_parse(inp)
