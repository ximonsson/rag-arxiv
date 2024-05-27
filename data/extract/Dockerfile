FROM python:3.9.19-alpine3.20

RUN pip install kaggle

WORKDIR /app
COPY ./extract.sh .

ENTRYPOINT ["sh", "extract.sh"]
