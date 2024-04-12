import streamlit as st
import db
import torch
import polars as pl
import plotly.graph_objects as go
import duckdb
import umap
import sklearn.cluster

torch.manual_seed(42)
con = duckdb.connect("/home/ximon/data/rag.db", read_only=True)

st.set_page_config(layout="centered")

st.title("Simon's Super Awesome Fantastic RAG System")
st.caption("Arxiv abstracts.")


@st.cache_resource
def ld_df():
    return con.sql("SELECT * FROM document").pl()


@st.cache_resource
def ld_embs():
    return torch.load("/home/ximon/data/rag/embeddings.pt")


@st.cache_resouce
def dim_reduce(x, d):
    return umap.UMAP(n_neighbors=15, n_components=d, metric="cosine").fit_transform(x)


@st.cache_resource
def cluster(x):
    return sklearn.cluster.HDBSCAN(min_cluster_size=15).fit(x)


df = ld_df()
embs = ld_embs()

# U, S, V = torch.pca_lowrank(embs)
# embs_pca = embs @ V[:, :3]

clusters = cluster(dim_reduce(embs, 5))
embs_3d = dim_reduce(embs, 3)

st.write(f"{embs.shape[-1]} documents in database")


def plot(prompt):
    color = (
        db.distance(db.embed([prompt]), embs)
        if prompt != ""
        else torch.zeros(embs.shape[0])
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=embs_3d[:, 0],
                y=embs_3d[:, 1],
                z=embs_3d[:, 2],
                mode="markers",
                text=df["title"][: embs.shape[0]],
                marker=dict(
                    size=1,
                    colorscale="reds",
                    color=color,
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


def search():
    _, i = db.search(prompt, df["document"].to_list(), embs, n_retrieve, 5)
    return df[i.numpy()]


n_retrieve = st.slider("Number of documents to retrieve", 0, 200, 20)
prompt = st.text_input("What would you want to ask?")


if prompt != "":
    docs = search()
    answer = db.generate(prompt, docs["document"].to_list())
    st.markdown(f"`> {answer['answer']}`")
    st.divider()
    st.dataframe(docs[:, ["id", "title", "authors"]].to_pandas())
    # plot our data
    st.plotly_chart(plot(prompt), use_container_width=True)
