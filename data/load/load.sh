FILENAME=arxiv-metadata-oai-snapshot.json
FILE=/tmp/$FILENAME

# only extract if file does not exist already?
if [ ! -e $FILE ]; then
	unzip $HOME/data/arxiv.zip -d /tmp/
fi

DB=":memory:"
ARXIV_RAW_JSON_FP=$FILE EXPORT_DST=$HOME/data/arxiv.parquet duckdb $DB < data/load/load.sql
