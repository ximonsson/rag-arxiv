#
# Extract raw data.
# ---
#
# Downloads the data from Kaggle, and takes as argument the path where to store the data.
#

kaggle datasets download -p $ARXIV_RAW_DIR -d Cornell-University/arxiv
