import os, time, io
from typing import List
from Bio import Entrez, SeqIO
from http.client import IncompleteRead
from urllib.error import HTTPError
import sqlite3
import pandas as pd
from typing import *

class EntrezQueries:
  """
    Encapsulates a set of predefined Entrez query strings.
    Instantiating or calling this class returns the full list of queries."""
  def __init__(self):
    # Predefined Entrez‐style queries for DNA database construction
    self.queries = [
      "BRCA1[Gene] AND Homo sapiens[Organism]",
      "TP53[Gene] AND Homo sapiens[Organism]",
      "CFTR[Gene] AND Homo sapiens[Organism]",
      "G6PD[Gene] AND Homo sapiens[Organism]",
      "MTHFR[Gene] AND Homo sapiens[Organism]",
      "HBB[Gene] AND Homo sapiens[Organism]",
      "PAH[Gene] AND Homo sapiens[Organism]",
      "MT-CO1[Gene] AND Homo sapiens[Organism]",
      "COX3[Gene] AND Mus musculus[Organism]",
      "BRCA2[Gene] AND Mus musculus[Organism]",
      "SOD1[Gene] AND Mus musculus[Organism]",
      "CFTR[Gene] AND Pan troglodytes[Organism]",
      "hemoglobin[Gene] AND Gallus gallus[Organism]",
      "cytochrome b[Gene] AND Drosophila melanogaster[Organism]",
      "ribosomal protein L3[Gene] AND Saccharomyces cerevisiae[Organism]",
      "recA[Gene] AND Escherichia coli[Organism]",
      "rpoB[Gene] AND Mycobacterium tuberculosis[Organism]",
      "16S rRNA[Gene] AND Streptomyces coelicolor[Organism]",
      "COI[Gene] AND Danio rerio[Organism]",
      "PSEN1[Gene] AND Homo sapiens[Organism]",
      "KRAS[Gene] AND Homo sapiens[Organism]",
      "EGFR[Gene] AND Homo sapiens[Organism]",
      "PDGFRA[Gene] AND Rattus norvegicus[Organism]",
      "rbcL[Gene] AND Arabidopsis thaliana[Organism]",
      "chlorophyll a/b binding protein[Gene] AND Oryza sativa[Organism]",
      "ACTB[Gene] AND Homo sapiens[Organism]",
      "MYC[Gene] AND Homo sapiens[Organism]",
      "FBN1[Gene] AND Homo sapiens[Organism]",
      "COL1A1[Gene] AND Homo sapiens[Organism]",
      "VHL[Gene] AND Homo sapiens[Organism]",
      "TTR[Gene] AND Homo sapiens[Organism]",
      "ATP7B[Gene] AND Homo sapiens[Organism]",
      "MLH1[Gene] AND Homo sapiens[Organism]",
      "APOE[Gene] AND Homo sapiens[Organism]",
      "HTT[Gene] AND Homo sapiens[Organism]",
      "ABCC6[Gene] AND Homo sapiens[Organism]",
      "CDKN2A[Gene] AND Homo sapiens[Organism]",
      "PAX6[Gene] AND Homo sapiens[Organism]",
      "SCN1A[Gene] AND Homo sapiens[Organism]",
      "MECP2[Gene] AND Homo sapiens[Organism]",
      "NOD2[Gene] AND Homo sapiens[Organism]",
      "ALB[Gene] AND Homo sapiens[Organism]",
      "INS[Gene] AND Homo sapiens[Organism]",
      "HLA-DQA1[Gene] AND Homo sapiens[Organism]",
      "TNF[Gene] AND Homo sapiens[Organism]",
      "FOXP2[Gene] AND Homo sapiens[Organism]",
      "CLOCK[Gene] AND Homo sapiens[Organism]",
      "CYP2D6[Gene] AND Homo sapiens[Organism]",
      "NR3C1[Gene] AND Homo sapiens[Organism]",
      "LCT[Gene] AND Homo sapiens[Organism]",
      "MEL1A[Gene] AND Homo sapiens[Organism]",
      "ACE2[Gene] AND Homo sapiens[Organism]",
      "SLC6A4[Gene] AND Homo sapiens[Organism]",
      "OXTR[Gene] AND Homo sapiens[Organism]",
      "TAS2R38[Gene] AND Homo sapiens[Organism]",
      "HNF4A[Gene] AND Homo sapiens[Organism]",
      "SHH[Gene] AND Mus musculus[Organism]",
      "OPN1LW[Gene] AND Homo sapiens[Organism]",
      "MYH7[Gene] AND Homo sapiens[Organism]",
      "TUBA1A[Gene] AND Homo sapiens[Organism]"
    ]

  def __call__(self):  return self.queries
  def __iter__(self):  return iter(self.queries)

class Database:
  """
    Automate DNA dataset collection & processing.
      Fetch raw FASTA from NCBI with guaranteed API key, immediate write
    and on-disk SeqIO index for O(1) lookup.
    Args:
      topics (List[str]): List containing strings of topics needed for building database
      out_dir (str) : path to output directory.
      email (str, optional): email address required by NCBI. Defaults to None.
      api_key (str, optional): NCBI API key for higher rate limits. Defaults to None.
      max_rate (float): Max requests in a second. needs `api_key`, `email` for `max_rate` > 3
      batch_size (int): Number of sequences per file. Defaults to 500.
      retmax (int): Max number of records to retrieve. Defaults to 10000.
      db (str): NCBI database to search. Defaults to `nucleotide`.
      fmt (str): fetching format for the Search. defaults to `fasta`.
  """

  def __init__(self, topics: List[str], out_dir: str, email: str, api_key: str, max_rate: float = 10.0, batch_size: int = 500, retmax: int = 10000, db: str = 'nucleotide', fmt: str = 'fasta'):
    if not api_key:
      raise ValueError("An NCBI API key is required.")
    if not email:
      raise ValueError("An Email is required.")
    Entrez.email   = email
    Entrez.api_key = api_key

    self.topics, self.out_dir = topics, out_dir
    self.max_rate, self.batch_size, self.retmax = max_rate, batch_size, retmax
    self.db, self.fmt = db, fmt
    self._sleep = 1.0 / self.max_rate
    os.makedirs(self.out_dir, exist_ok=True)

  def _sanitize(self, s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s).strip("_")

  def search(self, query: str) -> List[str]:
    handle = Entrez.esearch(db=self.db, term=query, retmax=self.retmax)
    rec = Entrez.read(handle)
    handle.close()
    return rec.get('IdList', [])

  def _safe_efetch(self, ids: List[str]) -> io.StringIO:
    for attempt in range(3):
      try:
        h = Entrez.efetch(db=self.db, id=",".join(ids), rettype=self.fmt, retmode='text')
        data = h.read()
        h.close()
        return io.StringIO(data)
      except (IncompleteRead, HTTPError) as e:
        print(f"[Warning] fetch attempt {attempt+1} failed: {e}")
        time.sleep(2 ** attempt)
    raise RuntimeError("Failed to fetch batch after retries")

  def build(self, with_index: bool=True):
    """
      For each topic:
        1. search -> get UIDs
        2. fetch in batches, write as you go
        3. build an on-disk index for O(1) lookups
    """
    for topic in self.topics:
      print(f"[+] Raw build: {topic}")
      ids = self.search(topic)
      print(f"\t-> Found {len(ids)} IDs")
      if not ids: continue

      fname = self._sanitize(topic)
      fasta_path = os.path.join(self.out_dir, f"{fname}.fasta")
      idx_path = os.path.join(self.out_dir, f"{fname}.idx")

      # open output file once
      with open(fasta_path, 'w', encoding='utf-8') as out_fh:
        for i in range(0, len(ids), self.batch_size):
          batch = ids[i:i + self.batch_size]
          try:
            handle = self._safe_efetch(batch)
            # stream write
            for line in handle:
              out_fh.write(line)
            out_fh.flush()
          except Exception as e:
            print(f"\t[Error] batch {i}-{i+len(batch)} failed: {e}")
            continue
          time.sleep(self._sleep)

      # build on‐disk index for O(1) record lookup
      if with_index:
        try:
          SeqIO.index_db(idx_path, fasta_path, "fasta")
          print(f"\t->Built index: {idx_path}")
        except Exception as e:
          print(f"\t[Warning] could not build index: {e}")
        print(f"\t->Completed raw FASTA: {fasta_path}\n")

def create_index(input_dir: str, index_path: str = "combined.idx"):
  """
    Build a single O(1) lookup index (.idx) from all FASTA files in a directory,
  skipping duplicate record IDs automatically.

  Args:
    input_dir:  Directory containing .fasta/.fa files.
    index_path: Path to SQLite index file to create.
  """
  # Gathering FASTA files
  fasta_files = [
    os.path.join(input_dir, fn)
    for fn in os.listdir(input_dir)
    if fn.lower().endswith(('.fa','fasta'))
  ]
  if not fasta_files:
    print(f"[!] No FASTA files found in `{input_dir}`.")
    return

  # Creating SQLite DB and table
  conn = sqlite3.connect(index_path)
  c = conn.cursor()
  c.execute("""
    CREATE TABLE IF NOT EXISTS seq_index (
      key      TEXT PRIMARY KEY,
      filename TEXT,
      offset   INTEGER,
      length   INTEGER
    )
  """)
  conn.commit()

  # scanning each FASTA and record offsets
  for fasta in fasta_files:
    print(f"[+] Indexing {fasta}")
    with open(fasta, 'r', encoding='utf-8') as fh:
      while True:
        pos = fh.tell()
        header = fh.readline()
        if not header: break
        if not header.startswith('>'): continue
        seq_id = header[1:].split()[0]
        # read sequence lines until next '>' or EOF
        seq_len = 0
        while True:
          line_start = fh.tell()
          line = fh.readline()
          if not line or line.startswith('>'):
            # rewind if we overshot onto next header
            if line and line.startswith('>'):
              fh.seek(line_start)
            break
          seq_len += len(line.strip())

        # Inserting or ignoring duplicates
        c.execute("INSERT OR IGNORE INTO seq_index (key,filename,offset,length) VALUES (?,?,?,?)", (seq_id, fasta, pos, seq_len))
    conn.commit()
  conn.close()
  print(f"[/] Built combined index at `{index_path}`")

def convert_fasta(input_dir: str, output_dir: str, mode: str = 'csv'):
  """
    Read all FASTA files in `input_dir` and write out CSV or Parquet files
    with columns: id, name (description), length (original), sequence.

    Args:
      input_dir:  Path to folder containing .fasta files.
      output_dir: Path where output files will be saved.
      mode: 'csv' or 'parquet'.
  """
  os.makedirs(output_dir, exist_ok=True)
  for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.fasta', '.fa')):
      continue
    path_in = os.path.join(input_dir, fname)
    recs = list(SeqIO.parse(path_in, 'fasta'))
    if not recs: continue
    rows = []
    for r in recs:
      seq_str = str(r.seq)
      rows.append({
        'id': r.id,
        'name': r.description,
        'length': len(seq_str),
        'sequence': seq_str })

    df = pd.DataFrame(rows)
    base = os.path.splitext(fname)[0]
    if mode == 'csv':
      out_path = os.path.join(output_dir, f"{base}.csv")
      df.to_csv(out_path, index=False)
    elif mode == 'parquet':
      out_path = os.path.join(output_dir, f"{base}.parquet")
      df.to_parquet(out_path, index=False)
    else:
      raise ValueError(f"Unsupported mode: {mode}")

    print(f"Converted {path_in} -> {out_path}")