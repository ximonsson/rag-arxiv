import db
import torch
import duckdb

con = duckdb.connect("/home/ximon/data/rag.db", read_only=True)
N = con.sql("SELECT count(*) FROM document").fetchall()[0][0]
BS = 65536


def embed(off: int):
    stmt = f"SELECT body FROM document LIMIT {BS} OFFSET {off}"
    print(stmt)
    rel = con.sql(stmt)
    return db.ingest(rel.pl()["body"].to_list())


embs = torch.cat([embed(i) for i in torch.arange(0, end=N, step=BS)], dim=0)
torch.save(embs, "/home/ximon/data/rag/embeddings.pt")
