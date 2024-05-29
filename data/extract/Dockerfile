FROM python:3.12-alpine

RUN pip install kaggle

WORKDIR /app
COPY ./extract.sh .

ENV ARXIV_RAW_DIR
ENV KAGGLE_USERNAME
ENV KAGGLE_KEY

CMD ["sh", "extract.sh"]
