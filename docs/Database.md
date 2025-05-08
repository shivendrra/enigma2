# NCBI Database Builder

A Python utility to **retrieve**, **align**, and **store** DNA sequence datasets from NCBI, organized by user-defined topics (e.g., genes, organisms). The core class is in `database.py`.

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration Details](#configuration-details)
* [Project Structure](#project-structure)
* [Technical Specifications](#technical-specifications)

## Features

* **Flexible Queries**: Search NCBI with any Entrez‐compatible term via `topics` list.
* **Batch Fetch & Rate Limiting**: Configurable `batch_size`, `max_rate`, and `retmax` parameters to handle large datasets responsibly.
* **Alignment**: Pads sequences to uniform length while preserving original lengths in metadata.
* **Multiple Output Formats**: Export aligned datasets as raw text, CSV, or Parquet.
* **Split/Merge Utilities**: Helper methods to split a FASTA into individual sequence files or merge sequences into one file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shivendrra/enigma2.git
   cd enigma2
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. Install dependencies:

   ```bash
   pip install biopython pandas pyarrow
   ```

## Usage

### Direct Invocation via `database.py`

Run the full database build process directly:

```bash
python database.py \
  --topics "BRCA1[Gene] AND Homo sapiens[Org]" "TP53[Gene] AND Homo sapiens[Org]" \
  --out_dir ./db_output \
  --mode csv \
  --email you@example.com \
  --api_key YOUR_NCBI_KEY \
  --max_rate 2.0 \
  --batch_size 200 \
  --retmax 5000
```

This generates aligned CSV files (one per topic) under `./db_output/`.

### Invocation via `test.py`

A minimal script (`test.py`) is provided for quick testing. Example content:

```python
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

from enigma2 import Database

db = Database(
  topics=["Human"],
  out_dir="./data/",
  mode='csv', # csv | text | paraquet
  email="your@email.com",
  retmax=1000
)
db.build()
```

Run:

```bash
python test.py
```

This retrieves up to 1000 Human nucleotide records, aligns them, and writes `./data/Human.csv`.

## Configuration Details

| Flag           | Description                                | Default      |
| -------------- | ------------------------------------------ | ------------ |
| `--topics`     | Entrez queries (list of strings)           | *required*   |
| `--out_dir`    | Base directory for outputs                 | *required*   |
| `--mode`       | Output format: `text`, `csv`, or `parquet` | `text`       |
| `--email`      | User email for NCBI Entrez                 | `None`       |
| `--api_key`    | NCBI API key (for higher rate limits)      | `None`       |
| `--max_rate`   | Maximum requests per second                | `3.0`        |
| `--batch_size` | Number of sequences per fetch batch        | `500`        |
| `--retmax`     | Maximum IDs returned per search            | `10000`      |
| `--db`         | Entrez database (e.g., `nucleotide`)       | `nucleotide` |
| `--format`     | EFetch format (e.g., `fasta`, `gb`)        | `fasta`      |

## Project Structure

```text
├── docs/
├── ├── Database.md
├── ├── Model.md
├── ├── Training.md
├── ├── User.md
├── enigma/
├── ├── config.json
├── ├── database.py    # Core Database class and CLI
├── ├── EnBert.py      # BERT model based on MoE architecture
├── ├── model.py       # Main enigma model
├── ├── run.py         # training code logic
├── ├── dataset.py     # dataset class to create training datasets & batches
├── README.md
├── requirements.txt  # List of Python dependencies
└── data/             # Example output directory
```

## Technical Specifications

* **Language**: Python 3.13+
* **Entrez Access**: Uses Biopython's `Entrez.esearch` and `Entrez.efetch`.
* **Data Handling**: pandas DataFrame → text/CSV/Parquet via `save_aligned()`.
* **Sequence Alignment**: Simple global padding implemented in `align_topic()`.
* **Metadata**: Original sequence lengths stored in `SeqRecord.annotations['original_length']`.
