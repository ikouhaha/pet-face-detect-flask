FROM python:3.8 as development

RUN apt-get update && apt-get install -y tini

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app


CMD [ "python", "app.py"]

FROM python:3.8 as production

RUN apt-get update && apt-get install -y tini

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app


CMD [ "python", "app.py"]

