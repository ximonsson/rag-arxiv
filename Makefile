NAMESPACE=k8s.io

extract: data/extract/*
	nerdctl -n $(NAMESPACE) build data/extract -t rag-arxiv_extract

load: data/load/*
	nerdctl -n $(NAMESPACE) build data/load -t rag-arxiv_load

transform: data/transform/*
	nerdctl -n $(NAMESPACE) build data/transform -t rag-arxiv_transform

data: extract load transform
