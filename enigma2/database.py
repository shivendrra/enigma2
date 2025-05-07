import os, time, re, csv, io
from typing import List, Optional
from Bio import Entrez, SeqIO
from http.client import IncompleteRead
from urllib.error import HTTPError
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
      "chlorophyll a/b binding protein[Gene] AND Oryza sativa[Organism]"
    ]

  def __call__(self):  return self.queries
  def __iter__(self):  return iter(self.queries)

class Database:
  """
    Automate DNA dataset collection, processing, and alignment into
    topic-specific “databases.”

    Args:
      topics (List[str]): List containing strings of topics needed for building database
      out_dir (str) : path to output directory.
      mode (str) [ 'text' | 'csv' | 'parquet' ]: mode to set the database in format.
      email (str, optional): email address required by NCBI. Defaults to None.
      api_key (str, optional): NCBI API key for higher rate limits. Defaults to None.
      max_rate (float): Max requests in a second. needs `api_key`, `email` for `max_rate` > 3
      batch_size (int): Number of sequences per file. Defaults to 500.
      retmax (int): Max number of records to retrieve. Defaults to 10000.
      db (str): NCBI database to search. Defaults to `nucleotide`.
      fmt (str): fetching format for the Search. defaults to `fasta`.
  """
  def __init__(self, topics: List[str], out_dir: str, mode: str = 'text', email: Optional[str] = None, api_key: Optional[str] = None, max_rate: float = 3.0, batch_size: int = 500, retmax: int = 10000, db: str = 'nucleotide', fmt: str = 'fasta', raw: bool = False):
    self.topics, self.out_dir, self.mode = topics, out_dir, mode
    self.batch_size = batch_size
    self.retmax = retmax
    self.db, self.fmt = db, fmt
    self.raw = raw

    if email:
      Entrez.email = email
    if api_key:
      Entrez.api_key = api_key

    os.makedirs(self.out_dir, exist_ok=True)
    self._sleep = 1.0 / max_rate  # NCBI allows up to max_rate req/sec -> sleep interval

  def _sanitize(self, s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', s).strip('_')

  def search(self, query: str) -> List[str]:
    handle = Entrez.esearch(db=self.db, term=query, retmax=self.retmax)
    rec = Entrez.read(handle)
    handle.close()
    return rec.get('IdList', [])

  def _safe_efetch(self, id_list):
    for _ in range(3):  # retry up to 3 times
      try:
        handle = Entrez.efetch(db=self.db, id=','.join(id_list), rettype=self.fmt, retmode='text')
        data = handle.read()
        handle.close()
        return io.StringIO(data)
      except (IncompleteRead, HTTPError) as e:
        print(f"Retrying after network error: {e}")
        time.sleep(2)
    raise RuntimeError("Failed to fetch after retries.")

  def fetch_raw(self, ids: List[str], topic: str) -> str:
    raw_dir = os.path.join(self.out_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{self._sanitize(topic)}.fasta")
    with open(path, 'w', encoding='utf-8') as out:
      for i in range(0, len(ids), self.batch_size):
        batch = ids[i:i+self.batch_size]
        h = Entrez.efetch(db=self.db, id=','.join(batch), rettype=self.fmt, retmode='text')
        out.write(h.read())
        h.close()
        time.sleep(self._sleep)
    return path

  def _determine_max_len(self, ids: List[str]) -> int:
    max_len = 0
    for i in range(0, len(ids), self.batch_size):
      batch = ids[i:i+self.batch_size]
      handle = self._safe_efetch(batch)
      for rec in SeqIO.parse(handle, 'fasta'):
        length = len(rec.seq)
        if length > max_len:
          max_len = length
      handle.close()
      time.sleep(self._sleep)
    return max_len

  def _stream_and_write(self, ids: List[str], max_len: int, topic: str):
    fname = self._sanitize(topic)
    out_path = os.path.join(self.out_dir, f"{fname}.{self.mode}")

    if self.mode == 'text':
      out_handle = open(out_path, 'w', encoding='utf-8')
    elif self.mode == 'csv':
      out_handle = open(out_path, 'w', newline='', encoding='utf-8')
      writer = csv.writer(out_handle)
      writer.writerow(['id','name','length','sequence'])
    elif self.mode == 'parquet':
      rows_buffer = []
      part_index  = 0
    else:
      raise ValueError(f"Unknown mode: {self.mode}")

    for i in range(0, len(ids), self.batch_size):
      batch = ids[i:i+self.batch_size]
      h = Entrez.efetch(db=self.db, id=','.join(batch), rettype=self.fmt, retmode='text')
      for rec in SeqIO.parse(h, 'fasta'):
        seq_str = str(rec.seq)
        padded = seq_str.ljust(max_len, '-')
        row = [rec.id, rec.description, len(seq_str), padded]
        if self.mode == 'text':
          out_handle.write(padded + '\n')
        elif self.mode == 'csv':
          writer.writerow(row)
        else:  # parquet
          rows_buffer.append({
            'id': rec.id,
            'name': rec.description,
            'length': len(seq_str),
            'sequence': padded})
          if len(rows_buffer) >= 10000:
            df = pd.DataFrame(rows_buffer)
            df.to_parquet(f"{out_path}.part{part_index}", index=False)
            part_index += 1
            rows_buffer.clear()
      h.close()
      time.sleep(self._sleep)

    if self.mode == 'parquet' and rows_buffer:
      df = pd.DataFrame(rows_buffer)
      df.to_parquet(f"{out_path}.part{part_index}", index=False)
    out_handle.close()
    print(f"\t>> Wrote streamed {self.mode} database: {out_path}")

  def build_raw(self):
    """
      For each topic: perform ESearch → EFetch batches → write raw FASTA """
    for topic in self.topics:
      print(f"[+] Raw build for topic: {topic}")
      ids = self.search(topic)
      print(f"\t>> Found {len(ids)} IDs")
      if not ids:
        continue
      fasta_path = self.fetch_raw(ids, topic)
      print(f"\t>> Wrote raw FASTA → {fasta_path}")

  def build_aligned_sequence(self):
    """
      For each topic: perform ESearch → determine max length → stream-align & write"""
    for topic in self.topics:
      print(f"[+] Aligned build for topic: {topic}")
      ids = self.search(topic)
      print(f"\t>> Found {len(ids)} IDs")
      if not ids:
        continue

      max_len = self._determine_max_len(ids)
      print(f"\t>> Determined max sequence length: {max_len}")

      self._stream_and_write(ids, max_len, topic)
      print(f"\t>> Completed aligned database for '{topic}'")

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