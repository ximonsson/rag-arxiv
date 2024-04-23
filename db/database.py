import duckdb
import transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
import torch
from typing import Union
import numpy as np
from tqdm import tqdm
import polars as pl
import hdbscan
import umap
import yaml


class Database:
    """
    Implements a simple vector database.
    """

    def __ld_models__(
        self, retrieval: str = "", generative: str = "", crossencoder: str = ""
    ):
        # Retrieval Model
        self.retrieval_tokenizer = AutoTokenizer.from_pretrained(retrieval)
        self.retrieval_model = AutoModel.from_pretrained(retrieval)

        # Cross encoder model
        self.xenc_tokenizer = AutoTokenizer.from_pretrained(crossencoder)
        self.xenc_model = AutoModelForSequenceClassification.from_pretrained(
            crossencoder
        )

        # Generative model
        self.gen_model = transformers.pipeline(
            "question-answering", model=generative, tokenizer=generative
        )

    def __ld_storage__(self, backend: str = "duckdb", path: Union[str, None] = None):
        self.quack = duckdb.connect(path)

    def __init__(self, **kwargs):
        self.__ld_models__(**kwargs["models"])
        self.__ld_storage__(**kwargs["storage"])

    def from_config_file(fp: str) -> "Database":
        """
        Create new Database instance from config file path.
        """

        with open(fp) as f:
            return Database(**yaml.safe_load(f))

    def __embed__(self, doc: Union[str, list[str]]) -> torch.Tensor:
        x = self.retrieval_tokenizer(
            doc, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            y = self.retrieval_model(**x)

        # make sure to take into account attention mask when calculating mean to make sure
        # it is a correct representation.

        mask = x["attention_mask"].unsqueeze(-1).expand(y[0].shape)
        return torch.sum(y[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def retrieve(self, q: str, n: int) -> pl.DataFrame:
        """
        Retrieve documents that are most similar to query.
        """

        q_ = self.__embed__(q)
        return self.quack.execute(
            """SELECT * FROM doc ORDER BY list_cosine_similiarity(?, embedding) DESC LIMIT ?""",
            q_.numpy(),
            n,
        ).pl()

    def rerank(self, q: str, doc: list[str]) -> tuple:
        """
        Use cross encoder to rerank the documents that were previously retrieved.
        """

        x = self.xenc_tokenizer(
            [q] * len(docs), docs, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            y = self.xenc_model(**x).logits.flatten()

        return torch.topk(y, n)

    def search(self, q: str, nret: int = 20, nx: int = 5) -> pl.DataFrame:
        """
        Document search for the most relevant documents.
        """

        docs = self.retrieve(q, db, nret)
        s, j = self.rerank(q, docs["body"].to_list(), nx)
        return docs[j]

    def __gen__(self, q: str, docs: list[str]) -> str:
        """
        Generate answer to prompt based on context.
        """

        return self.gen_model({"question": q, "context": "\n".join(docs)})

    def query(self, q: str) -> str:
        """
        Query information from the documents.
        """

        docs = self.search(q)
        return self.__gen__(q, docs["body"])

    def __dim_reduce__(x: torch.Tensor, d: int) -> torch.Tensor:
        return torch.Tensor(
            umap.UMAP(n_neighbors=15, n_components=d, metric="cosine").fit_transform(x)
        )

    def __cluster__(self, x: torch.Tensor) -> np.ndarray:
        """
        Cluster the documents in the database.
        """

        y = (
            hdbscan.HDBSCAN(min_cluster_size=15)
            .fit(Database.__dim_reduce__(x, 5))
            .labels_
        )
        return y

    def __topics__(self):
        """
        Generate topics for the clusters.
        """

        pass

    def ingest(self, doc: Union[duckdb.duckdb.DuckDBPyRelation, pl.DataFrame]):
        """
        Ingest a set of documents to the database.

        This will retrigger clustering and topic modelling.
        """

        data = self.quack.sql("SELECT body FROM doc").list("body").fetchall()[0][0]

        # embed all the documents
        bs = 1  # TODO this is only on CPU though!
        ys = [
            self.__embed__(data[i : i + bs])
            for i in tqdm(torch.arange(0, len(data), step=bs))
        ]
        ys = torch.cat(ys, 0)
        embedding = pl.DataFrame(pl.Series("embedding", ys.numpy()))

        # cluster documents
        clusters = self.__cluster__(ys)
        topic = pl.DataFrame(pl.Series("topic", clusters))

        self.quack.sql(
            """
        CREATE OR REPLACE TABLE doc AS
            SELECT doc.*, embedding.*, topic.*
            FROM doc POSITIONAL JOIN embedding POSITIONAL JOIN topic
        """
        )

        # generate topics for the clusters
        self.__topics__()

    def docs(self) -> duckdb.duckdb.DuckDBPyRelation:
        """
        Return documents that are stored in the database.
        """

        return self.quack.sql("SELECT * FROM doc")

    def sql(self, stmt: str) -> duckdb.duckdb.DuckDBPyRelation:
        """
        PRO MODE.
        """

        return self.quack.sql(stmt)
