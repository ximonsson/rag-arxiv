#
# Extract raw data.
# ---
# Downloads the data from Kaggle, and takes as argument the path where to store the data.
#

if [[ $# < 1 ]]; then
	echo "please specify download path!"
	exit 1
elif [[ ! -d $1 ]]; then
	echo "dir '$1' does not exist!"
	exit 1
fi

DST=$1
DATASET=Cornell-University/arxiv
kaggle datasets download -p $DST -d $DATASET
