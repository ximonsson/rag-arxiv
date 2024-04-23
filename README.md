# Document Search/RAG System For Articles Published in arXiv

Small learning project in how to create a document search / RAG system for articles that are published in [arxiv](https://arxiv.org/). I decided to everything from scratch for my own curiousity.


## Setup

There are some environment variables that need to be set for the system to work.

- `ARXIV_RAW_DIR` - where to store the raw data from the extraction step.
- `ARXIV_CLEAN_FP` - where to store the cleaned file from the load step.
- `ARXIV_DOC_FP` - where to store the documents from the transform step.


## Data: ELT Pipeline

### Extraction

Raw data is fetched from [kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) using the [Kaggle Public API](https://github.com/Kaggle/kaggle-api#api-credentials). You will need to get an API token store it at `$HOME/.kaggle/kaggle.json`. For more information see [here](https://www.kaggle.com/docs/api) under **Getting Started: Installation & Authentication**.

Code is in [data/extraction](./data/extraction) directory. To run this step, run the [data/extraction/extraction.sh](./data/extraction/extraction.sh) script.

### Load

The raw JSON downloaded in the previous stepped is transformed into a tabular format and cleaned up, to later be stored as a parquet file.

### Transform

The relevant articles are filtered out and a column `body` is created in the format "{title}, written by {authors}, {abstract}" is created. This column is what is used for indexing later.


## Vector Database

The vector database implementation can be found in [db](./db). This only really implements functionality for a vector database, storage is handled by the client.

### Ingestion

### Clustering & Topic Modelling

### Querying


## Demo Application
