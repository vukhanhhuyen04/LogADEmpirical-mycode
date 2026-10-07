import csv
import json
import re
import numpy as np

DATA_DIR = "dataset/bgl/"
TEMPLATES_CSV = DATA_DIR + "BGL.log_templates.csv"
VEC_FILE = "dataset/nlp-word.vec"
OUT_FILE = DATA_DIR + "embeddings.json"
DIM = 300


def tokenize(template):
    template = template.replace("<*>", " ")
    words = re.split(r"[^a-zA-Z]+", template)
    return [w.lower() for w in words if w]


event_templates = {}
with open(TEMPLATES_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        event_templates[row["EventId"]] = tokenize(row["EventTemplate"])

needed_words = set()
for words in event_templates.values():
    needed_words.update(words)

print(f"{len(event_templates)} events, {len(needed_words)} unique words needed")

word_vectors = {}
with open(VEC_FILE, "r", encoding="utf-8", errors="ignore") as f:
    next(f)  # header line: vocab_size dim
    for line in f:
        parts = line.rstrip().split(" ")
        word = parts[0]
        if word in needed_words:
            word_vectors[word] = np.asarray(parts[1:1 + DIM], dtype=float)

print(f"found vectors for {len(word_vectors)}/{len(needed_words)} words")

embeddings = {}
zero_count = 0
for event_id, words in event_templates.items():
    vecs = [word_vectors[w] for w in words if w in word_vectors]
    if vecs:
        emb = np.mean(vecs, axis=0)
    else:
        emb = np.zeros(DIM)
        zero_count += 1
    embeddings[event_id] = emb.tolist()

print(f"{zero_count} events got zero vector (no known words)")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embeddings, f)

print("DONE, wrote", OUT_FILE)
