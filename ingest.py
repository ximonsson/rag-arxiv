import db
import torch
import duckdb
import os

DOC_FP = os.environ["ARXIV_DOC_FP"]

doc = duckdb.sql(f"SELECT * FROM read_parquet('{DOC_FP}')")
N = duckdb.execute("SELECT count(*) FROM doc").fetchall()[0][0]
BS = 65536


def embed(off: int):
    stmt = f"SELECT body FROM doc LIMIT {BS} OFFSET {off}"
    print(stmt)
    rel = duckdb.sql(stmt)
    return db.ingest(rel.list("body").fetchall()[0][0])


embs = torch.cat([embed(i) for i in torch.arange(0, end=N, step=BS)], dim=0)
torch.save(embs, os.environ["ARXIV_DOC_EMBEDDING_FP"])
