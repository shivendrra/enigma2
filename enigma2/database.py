import os, time, re
from typing import List, Optional
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd

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

  def __call__(self):
    """
      Returns the list of predefined queries."""
    return self.queries

  def __iter__(self):
    """
      Allows iteration over the queries."""
    return iter(self.queries)

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
  def __init__(self, topics: List[str], out_dir: str, mode: str = 'text', email: Optional[str] = None, api_key: Optional[str] = None, max_rate: float = 3.0, batch_size: int = 500, retmax: int = 10000, db: str = 'nucleotide', fmt: str = 'fasta'):
    self.topics, self.out_dir, self.mode = topics, out_dir, mode
    self.batch_size = batch_size
    self.retmax = retmax
    self.db, self.fmt = db, fmt

    if email:
      Entrez.email = email
    if api_key:
      Entrez.api_key = api_key

    os.makedirs(self.out_dir, exist_ok=True)
    self._sleep = 1.0 / max_rate  # NCBI allows up to max_rate req/sec -> sleep interval

  def _sanitize(self, s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', s).strip('_')

  def search(self, query: str) -> List[str]:
    """ESearch -> list of UIDs."""
    handle = Entrez.esearch(db=self.db, term=query, retmax=self.retmax)
    rec = Entrez.read(handle)
    handle.close()
    return rec.get('IdList', [])

  def fetch(self, ids: List[str], topic: str) -> str:
    """
      EFetch in batches -> single FASTA file per topic.
      Returns path to the FASTA file.
    """
    raw_dir = os.path.join(self.out_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f'{self._sanitize(topic)}.fasta')

    with open(out_path, 'w', encoding="utf-8") as out_handle:
      for i in range(0, len(ids), self.batch_size):
        batch = ids[i:i + self.batch_size]
        h = Entrez.efetch(db=self.db, id=','.join(batch), rettype=self.fmt, retmode='text')
        out_handle.write(h.read())
        h.close()
        time.sleep(self._sleep)
    return out_path

  def split(self, fasta_path: str, subdir: str = 'split'):
    """
      Split FASTA -> one file per sequence, named by description.
      Saves into out_dir/subdir.
    """
    outd = os.path.join(self.out_dir, subdir)
    os.makedirs(outd, exist_ok=True)
    for rec in SeqIO.parse(fasta_path, 'fasta'):
      name = self._sanitize(rec.description)
      p = os.path.join(outd, f'{name}.txt')
      with open(p, 'w', encoding="utf-8") as fh:
        fh.write(str(rec.seq))
  
  def merge(self, fasta_path: str, out_name: str = 'merged.txt'):
    """
      Merge FASTA -> single text file of raw DNA sequences, one per line.
    """
    outp = os.path.join(self.out_dir, out_name)
    with open(outp, 'w', encoding="utf-8") as out:
      for rec in SeqIO.parse(fasta_path, 'fasta'):
        out.write(str(rec.seq) + '\n')

  def align_topic(self, fasta_path: str) -> List[SeqRecord]:
    """
      Simple alignment: pads all sequences to maximal length with '-' at end.
      Returns a list of aligned SeqRecord.
    """
    records = list(SeqIO.parse(fasta_path, 'fasta'))
    original_lengths = {r.id: len(r.seq) for r in records}
    max_len = max(len(r.seq) for r in records)
    for r in records:
      padded = str(r.seq).ljust(max_len, '-')
      r.seq = Seq(padded)
      r.annotations['original_length'] = original_lengths[r.id]
    return records

  def save_aligned(self, aligned: List[SeqRecord], topic: str):
    """
      Save aligned sequences in chosen mode (text, csv, parquet).
    """
    data = {
      'id':       [r.id for r in aligned],
      'name':     [r.description for r in aligned],
      'length':   [r.annotations.get('original_length', len(r.seq)) for r in aligned],
      'sequence': [str(r.seq) for r in aligned]
    }
    df = pd.DataFrame(data)

    fname = f"{self._sanitize(topic)}"
    if self.mode == 'text':
      path = os.path.join(self.out_dir, f'{fname}.txt')
      df.sequence.to_csv(path, index=False, header=False)
    elif self.mode == 'csv':
      path = os.path.join(self.out_dir, f'{fname}.csv')
      df.to_csv(path, index=False)
    elif self.mode == 'parquet':
      path = os.path.join(self.out_dir, f'{fname}.parquet')
      df.to_parquet(path, index=False)
    else:
      raise ValueError(f"Unknown mode: {self.mode}")

  def build(self):
    """
      Full pipeline: for each topic -> search -> fetch -> align -> save.
    """
    for topic in self.topics:
      print(f"[+] Processing topic: {topic}")
      ids = self.search(topic)
      print(f"    >> Found {len(ids)} IDs")
      fasta = self.fetch(ids, topic)
      print(f"    >> Fetched FASTA -> {fasta}")
      aligned = self.align_topic(fasta)
      print(f"    >> Aligned {len(aligned)} sequences")
      self.save_aligned(aligned, topic)
      print(f"    >> Saved aligned database for '{topic}'\n")