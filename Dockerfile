FROM python:3.8.0-buster
MAINTAINER Evgeni Kasatkin
ENV TZ=Europe/Nicosia

RUN apt-get update

# Install Python dependencies.
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

COPY /app .
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 main:app




