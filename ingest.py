import db
import duckdb
import os

DOC_FP = os.environ["ARXIV_DOC_FP"]
doc = duckdb.sql(f"SELECT * FROM read_parquet('{DOC_FP}')")
DB.from_config_file("db.yml")
DB.ingest(doc)
