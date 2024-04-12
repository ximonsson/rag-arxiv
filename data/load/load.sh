DATADIR=$HOME/data
FILENAME=arxiv-metadata-oai-snapshot.json
FILE=$DATADIR/$FILENAME
DB=$DATADIR/rag.db

if [ ! -e $FILE ]; then
	unzip $DATADIR/arxiv.zip -d $DATADIR
fi

duckdb $DB < load.sql

#rm $FILE
