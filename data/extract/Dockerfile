FROM python:3.12-alpine

RUN pip install kaggle

WORKDIR /app
COPY ./extract.sh .

CMD ["sh", "extract.sh"]
