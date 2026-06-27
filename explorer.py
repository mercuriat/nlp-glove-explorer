"""
GloVe Word Similarity Explorer
--------------------------------
Loads pre-trained GloVe vectors and lets you interactively explore:
  - Nearest neighbours of any word
  - Cosine similarity between two words
  - Word analogy arithmetic (king - man + woman = ?)

"""

import numpy as np
from collections import defaultdict


# ── 1. Load GloVe vectors ──────────────────────────────────────────────────

def load_glove(path="glove.6B.50d.txt", max_words=50000):
    """
    Loads GloVe vectors from a text file into a dictionary.
    Each line is: word float float float ...
    We load max_words to keep memory manageable.
    """
    print(f"Loading GloVe vectors from {path}...")
    embeddings = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    print(f"Loaded {len(embeddings):,} word vectors ({vector.shape[0]} dimensions each)\n")
    return embeddings


# ── 2. Cosine similarity ───────────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """
    Measures the angle between two vectors.
    1.0  = identical direction (same meaning)
    0.0  = perpendicular (unrelated)
    -1.0 = opposite direction (antonyms)

    Formula: dot(A, B) / (|A| × |B|)
    """
    dot_product = np.dot(vec_a, vec_b)           # sum of products
    magnitude_a = np.linalg.norm(vec_a)          # √(sum of squares)
    magnitude_b = np.linalg.norm(vec_b)          # √(sum of squares)
    return dot_product / (magnitude_a * magnitude_b)


# ── 3. Find nearest neighbours ─────────────────────────────────────────────

def nearest_neighbours(word, embeddings, n=8):
    """
    Finds the n most similar words to the given word
    by computing cosine similarity against all loaded vectors.
    """
    if word not in embeddings:
        print(f"  '{word}' not found in vocabulary.\n")
        return []

    query_vec = embeddings[word]
    scores = []

    for other_word, other_vec in embeddings.items():
        if other_word == word:
            continue
        score = cosine_similarity(query_vec, other_vec)
        scores.append((other_word, score))

    # Sort by similarity, highest first
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# ── 4. Word analogy arithmetic ─────────────────────────────────────────────

def analogy(word_a, word_b, word_c, embeddings, n=5):
    """
    Computes: word_a - word_b + word_c = ?
    Example:  king   - man   + woman   = queen

    The result vector is compared against all words
    to find the closest match.
    """
    for w in [word_a, word_b, word_c]:
        if w not in embeddings:
            print(f"  '{w}' not found in vocabulary.\n")
            return []

    result_vec = embeddings[word_a] - embeddings[word_b] + embeddings[word_c]
    exclude = {word_a, word_b, word_c}

    scores = []
    for word, vec in embeddings.items():
        if word in exclude:
            continue
        score = cosine_similarity(result_vec, vec)
        scores.append((word, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n]


# ── 5. Pretty print helpers ────────────────────────────────────────────────

def print_neighbours(word, results):
    print(f"\n  Nearest neighbours of '{word}':")
    print(f"  {'Word':<15} {'Similarity':>10}  {'Bar'}")
    print(f"  {'─'*15} {'─'*10}  {'─'*20}")
    for w, score in results:
        bar = "█" * int(score * 20)
        print(f"  {w:<15} {score:>10.4f}  {bar}")
    print()

def print_analogy(a, b, c, results):
    print(f"\n  {a} − {b} + {c} = ?")
    print(f"  {'Word':<15} {'Similarity':>10}")
    print(f"  {'─'*15} {'─'*10}")
    for w, score in results:
        print(f"  {w:<15} {score:>10.4f}")
    print()


# ── 6. Interactive explorer ────────────────────────────────────────────────

def run_explorer(embeddings):
    print("=" * 50)
    print("  GloVe Word Similarity Explorer")
    print("  Commands:")
    print("    similar <word>           — nearest neighbours")
    print("    compare <word1> <word2>  — cosine similarity")
    print("    analogy <a> <b> <c>      — a - b + c = ?")
    print("    quit                     — exit")
    print("=" * 50)

    while True:
        try:
            raw = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0]

        if cmd == "quit":
            print("Goodbye!")
            break

        elif cmd == "similar" and len(parts) == 2:
            word = parts[1]
            results = nearest_neighbours(word, embeddings)
            if results:
                print_neighbours(word, results)

        elif cmd == "compare" and len(parts) == 3:
            w1, w2 = parts[1], parts[2]
            if w1 in embeddings and w2 in embeddings:
                score = cosine_similarity(embeddings[w1], embeddings[w2])
                print(f"\n  cosine_similarity('{w1}', '{w2}') = {score:.4f}\n")
            else:
                missing = [w for w in [w1, w2] if w not in embeddings]
                print(f"  Not found: {missing}\n")

        elif cmd == "analogy" and len(parts) == 4:
            a, b, c = parts[1], parts[2], parts[3]
            results = analogy(a, b, c, embeddings)
            if results:
                print_analogy(a, b, c, results)

        else:
            print("  Unknown command. Try: similar run | compare run sprint | analogy king man woman")


# ── 7. Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    embeddings = load_glove("glove.6B.50d.txt", max_words=50000)
    run_explorer(embeddings)