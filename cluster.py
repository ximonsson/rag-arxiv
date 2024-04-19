import duckdb
import torch
import hdbscan
import umap
import polars as pl

x = torch.load("/home/ximon/data/rag/embeddings.pt")
y = umap.UMAP(n_neighbors=15, n_components=5, metric="cosine").fit_transform(x)
c = hdbscan.HDBSCAN(min_cluster_size=15).fit(y).labels_

cluster = pl.DataFrame([pl.Series("cluster", c)])
con = duckdb.connect("/home/ximon/data/rag.db")
con.sql(
    "CREATE OR REPLACE TABLE document AS ("
    "SELECT document.*, cluster.* FROM document POSITIONAL JOIN cluster"
    ")"
)
