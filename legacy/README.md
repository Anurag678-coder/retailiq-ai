# Legacy

Not part of the default pipeline. Kept for reference only.

- `data_generation.py` — the original synthetic-data generator from the
  first version of this project. The current pipeline uses real transaction
  data (`src/download_data.py` + `src/preprocessing.py`), so this script is
  no longer imported or run anywhere.
- `notebooks/` — exploratory notebooks written against the old synthetic
  dataset. Column names and file paths in them are stale; they're kept only
  as a record of the original EDA process, not as working code.
