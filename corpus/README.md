Each processed corpus snapshot is written here as `version_vN/`.

Every version contains:
- `manifest.json`: file fingerprints from `data_lake/`
- `chunks.jsonl`: processed chunk payloads ready for retrieval
- `faiss_index/`: serialized FAISS vector index for that corpus version
