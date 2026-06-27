# GloVe Word Similarity Explorer

An interactive command-line tool for exploring word meaning through pre-trained GloVe word embeddings.


## What it does

- Find the **nearest neighbours** of any word by cosine similarity
- **Compare** two words and get their similarity score
- Solve **word analogies** using vector arithmetic (king − man + woman = queen)

## Concepts demonstrated

- Word embeddings — representing words as points in n-dimensional space
- Cosine similarity — measuring the angle between two vectors
- Vector arithmetic — encoding semantic relationships geometrically

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/mercuriat/nlp-glove-explorer.git
cd nlp-glove-explorer
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install numpy
```

### 4. Download GloVe vectors
Download `glove.6B.zip` from the [Stanford NLP page](https://nlp.stanford.edu/projects/glove/) and extract `glove.6B.50d.txt` into the project folder.

> The vector file is ~170MB and not included in this repo.

### 5. Run
```bash
python explorer.py
```

## Usage

```
> similar run
> compare market markets
> compare run jog
> analogy king man woman
```

## Example output

```
Nearest neighbours of 'run':
  Word            Similarity  Bar
  ─────────────── ──────────  ────────────────────
  running              0.8921  █████████████████
  runs                 0.8134  ████████████████
  sprint               0.7203  ██████████████
  ...
```

## Dependencies

- Python 3.8+
- numpy
- GloVe 6B 50d vectors (Stanford NLP)

## Author

MQA Porto