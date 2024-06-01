FILENAME=arxiv-metadata-oai-snapshot.json
FILE=/tmp/$FILENAME

# only extract if file does not exist already?
if [ ! -e $FILE ]; then
	unzip $ARXIV_RAW_DIR/arxiv.zip -d /tmp/
fi

DB=":memory:"
ARXIV_RAW_JSON_FP=$FILE duckdb $DB < load.sql
mv arxiv.parquet $ARXIV_CLEAN_FP
