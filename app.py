import streamlit as st
import db
import plotly.graph_objects as go
import duckdb
import sklearn.cluster
import os
import numpy as np


st.set_page_config(layout="centered")

st.title("Simon's Super Awesome Fantastic RAG System")
st.caption("Arxiv abstracts.")


@st.cache_resource
def ld():
    DB = db.Database.from_config_file("db.yml", read_only=True)
    embs = DB.embeddings
    embs_3d = db.Database.__dim_reduce__(embs, 3)  # used for plotting

    return DB, embs, embs_3d


DB, embs, embs_3d = ld()

st.write(f"{embs.shape[0]} documents in database")


def plot(prompt):
    N = embs.shape[0]  # all documents might not be embedded

    color = DB.distance(prompt) if prompt != "" else DB.doctopic
    titles = DB.docs()["title"].fetchnumpy()["title"]

    i = np.random.choice(range(embs_3d.shape[0]), 100000, replace=False)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=embs_3d[i, 0],
                y=embs_3d[i, 1],
                z=embs_3d[i, 2],
                mode="markers",
                text=titles[i].tolist(),
                marker=dict(
                    size=1,
                    colorscale="jet" if prompt == "" else "reds",
                    color=color[i],
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


n_retrieve = st.slider("Number of documents to retrieve", 0, 200, 20)
n_rerank = st.slider("Number of documents after reranking", 0, 200, 5)
prompt = st.text_input("What would you want to ask?")


def search():
    return DB.search(prompt, n_retrieve, n_rerank)


if prompt != "":
    # most relevant documents
    docs = search()

    # generate an answer
    answer = DB.query(prompt)
    st.markdown(f"`> {answer['answer']}`")
    st.divider()
    st.dataframe(docs[:, ["id", "title", "authors"]].to_pandas())

# plot our data
st.plotly_chart(plot(prompt), use_container_width=True)
