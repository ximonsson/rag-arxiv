import duckdb
import torch
import os
import db

DOC_FP = os.environ["ARXIV_DOC_FP"]
CLUSTER_FP = os.environ["ARXIV_DOC_CLUSTER_FP"]

X = torch.load(os.environ["ARXIV_DOC_EMBEDDING_FP"])
y = db.cluster(X)

duckdb.sql(
    f"""
    COPY (
        SELECT doc.eid, y.column0 AS cluster
        FROM read_parquet('{DOC_FP}') AS doc
        POSITIONAL JOIN y
    ) TO '{CLUSTER_FP}' (FORMAT 'parquet')
    """
)
