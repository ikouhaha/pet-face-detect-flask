FROM python:3.8

# RUN docker rm -vf $(docker ps -aq)
# RUN docker rmi -f $(docker images -aq)

RUN apt-get update && apt-get install -y tini

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 80 5000

CMD [ "python", "app.py"]