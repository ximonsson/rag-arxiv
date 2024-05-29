FROM debian:unstable-slim

WORKDIR /app

# install duckdb
RUN apt update && apt install -y wget unzip
RUN wget https://github.com/duckdb/duckdb/releases/download/v0.10.3/duckdb_cli-linux-amd64.zip
RUN unzip duckdb_cli-linux-amd64.zip
RUN mv duckdb /usr/bin

COPY load.sh .
COPY load.sql

ENV ARXIV_RAW_DIR
ENV ARXIV_CLEAN_FP

CMD ["sh", "load.sh"]
