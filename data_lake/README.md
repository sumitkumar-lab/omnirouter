Drop research papers and supporting notes directly in this folder.

Example layout:

data_lake/
  chinchilla_train.pdf
  chinchilla_train.md
  omnirouter_facts.txt

PDFs are indexed directly with the built-in PDF text extractor. For better math retrieval,
place a Nougat/Marker Markdown export next to the PDF with the same filename. The
pipeline will prefer that sidecar Markdown, preserve LaTeX text, and chunk by Markdown
headers such as Methodology, Proofs, Experiments, and Appendix.

Optional grouped folders are still supported when a real source group matters, but the
default upload path is the data_lake root:

data_lake/
  scaling_papers/
    chinchilla_train.pdf
    chinchilla_train.md

The ingestion pipeline scans this directory automatically and versions processed output under `corpus/`.
