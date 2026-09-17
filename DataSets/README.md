# DataSets

This folder is where the notebook caches its Gaia Archive query results. It starts empty on purpose.

The first run of `Rotation.ipynb` writes:

- `gaia_dr3_train.csv` and `gaia_dr2_test.csv`, the two samples
- `gaia_dr3_train.csv.adql` and `gaia_dr2_test.csv.adql`, the exact queries that produced them
- `query_log.json`, with the archive job URLs, timestamps, row counts and which query variant succeeded

These files are ignored by git, since they are large and fully reproducible. Delete them (or set `REDOWNLOAD = True`) to force a fresh query.

The Julia notebook in `pluto/` has an offline fallback that looks for `pluto/DataSets/legacy/*.csv`, the April 2025 files from the original project. Those files are not tracked here either. The fallback is disabled by default and is not needed for a normal run.
