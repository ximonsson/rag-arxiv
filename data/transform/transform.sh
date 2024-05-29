DB=":memory:"
duckdb < transform.sql
mv doc.parquet $ARXIV_DOC_FP
