FROM debian:unstable-slim

WORKDIR /app

# install duckdb
RUN apt update && apt install -y wget unzip
RUN wget https://github.com/duckdb/duckdb/releases/download/v0.10.3/duckdb_cli-linux-amd64.zip
RUN unzip duckdb_cli-linux-amd64.zip
RUN mv duckdb /usr/bin

COPY transform.sql .
COPY transform.sh .

ENV ARXIV_CLEAN_FP
ENV ARXIV_DOC_FP

CMD ["sh", "transform.sh"]
