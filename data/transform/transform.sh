DB=":memory:"
duckdb < data/transform/transform.sql
mv doc.parquet $ARXIV_DOC_FP
