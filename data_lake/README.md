Drop raw source files here under a source-specific folder.

Example layout:

data_lake/
  product_docs/
    architecture.pdf
    faq.json
  support_exports/
    incidents.csv

The ingestion pipeline scans this directory automatically and versions processed output under `corpus/`.
