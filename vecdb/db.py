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

# Models
# -------------------------

# Retrieval Model
retrieval_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
retrieval_tokenizer = AutoTokenizer.from_pretrained(retrieval_model_name)
retrieval_model = AutoModel.from_pretrained(retrieval_model_name)

# Cross encoder model
xenc_model_name = "cross-encoder/ms-marco-MiniLM-L-4-v2"
xenc_tokenizer = AutoTokenizer.from_pretrained(xenc_model_name)
xenc_model = AutoModelForSequenceClassification.from_pretrained(xenc_model_name)

# Generative model
gen_model_name = "deepset/tinyroberta-squad2"
gen_model = transformers.pipeline(
    "question-answering", model=gen_model_name, tokenizer=gen_model_name
)


def embed(docs: Union[str, list[str]]) -> torch.Tensor:
    x = retrieval_tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        y = retrieval_model(**x)

    # make sure to take into account attention mask when calculating mean to make sure
    # it is a correct representation.

    mask = x["attention_mask"].unsqueeze(-1).expand(y[0].shape)
    return torch.sum(y[0] * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def ingest(doc: list[str]) -> torch.Tensor:
    bs = 1  # batch size
    ys = [embed(doc[i : i + bs]) for i in tqdm(torch.arange(0, len(doc), step=bs))]
    return torch.cat(ys, 0)


# functions for searching
# ---


def distance(q: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
    return torch.cosine_similarity(q, db).flatten()


def retrieve(q: str, db: torch.Tensor, n: int) -> tuple:
    y = distance(embed(q), db)
    return torch.topk(y, n)


def rerank(q: str, docs: list[str], n: int) -> tuple:
    x = xenc_tokenizer(
        [q] * len(docs), docs, padding=True, truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        y = xenc_model(**x).logits.flatten()

    return torch.topk(y, n)


def search(
    q: str, docs: list[str], db: torch.Tensor, nret: int = 20, nx: int = 5
) -> torch.Tensor:
    _, i = retrieve(q, db, nret)
    # TODO don't like the converting to pl.Series and then back to list
    s, j = rerank(q, pl.Series(docs)[i.numpy()].to_list(), nx)
    return s, i[j]


def generate(q: str, docs: list[str]) -> str:
    return gen_model({"question": q, "context": "\n".join(docs)})


def query(q: str, docs: list[str], db: torch.Tensor) -> str:
    _, i = search(q, docs, db, 20, 5)
    return generate(q, docs[i.numpy()])


# topics and clustering
# ---


def dim_reduce(x: torch.Tensor, d: int) -> torch.Tensor:
    return torch.Tensor(
        umap.UMAP(n_neighbors=15, n_components=d, metric="cosine").fit_transform(x)
    )


def cluster(x: torch.Tensor) -> np.ndarray:
    return hdbscan.HDBSCAN(min_cluster_size=15).fit(dim_reduce(x, 5)).labels_
