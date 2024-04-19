import streamlit as st
import db
import torch
import plotly.graph_objects as go
import duckdb
import sklearn.cluster
import os

torch.manual_seed(42)

st.set_page_config(layout="centered")

st.title("Simon's Super Awesome Fantastic RAG System")
st.caption("Arxiv abstracts.")

DOC_FP = os.environ["ARXIV_DOC_FP"]
EMB_FP = os.environ["ARXIV_DOC_EMBEDDING_FP"]
CLUSTER_FP = os.environ["ARXIV_DOC_CLUSTER_FP"]


@st.cache_resource
def ld_doc():
    """Load documents and joins the clusters they have been assigned."""

    return duckdb.sql(
        f"""
    CREATE TABLE doc AS
        SELECT doc.*, cluster.* FROM
            (SELECT * FROM read_parquet('{DOC_FP}')) AS doc JOIN
            (SELECT * FROM read_parquet('{CLUSTER_FP}')) AS cluster ON
            cluster.eid = doc.eid
    """
    )


@st.cache_resource
def ld_embs():
    return torch.load(EMB_FP)


ld_doc()
embs = ld_embs()
embs_3d = db.dim_reduce(embs, 3)  # used for plotting

st.write(f"{embs.shape[-1]} documents in database")


def plot(prompt):
    N = embs.shape[0]  # all documents might not be embedded

    color = db.distance(db.embed([prompt]), embs) if prompt != "" else torch.zeros(N)
    titles = (
        duckdb.sql(f"SELECT title FROM doc LIMIT {N}").list("title").fetchall()[0][0]
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=embs_3d[:, 0],
                y=embs_3d[:, 1],
                z=embs_3d[:, 2],
                mode="markers",
                text=titles,
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
    docs = duckdb.sql("SELECT document FROM doc").fetchall()[0][0]
    _, i = db.search(prompt, docs, embs, n_retrieve, 5)
    return duckdb.execute(
        "SELECT * FROM doc WHERE rowid IN (SELECT unnest(?))", [i.numpy()]
    ).df()


n_retrieve = st.slider("Number of documents to retrieve", 0, 200, 20)
prompt = st.text_input("What would you want to ask?")


if prompt != "":
    docs = search()
    answer = db.generate(prompt, docs["document"].to_list())
    st.markdown(f"`> {answer['answer']}`")
    st.divider()
    st.dataframe(docs[:, ["id", "title", "authors"]])
    # plot our data
    st.plotly_chart(plot(prompt), use_container_width=True)
