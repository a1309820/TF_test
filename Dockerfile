FROM python:latest
ADD . /code
WORKDIR /code
RUN pip3 install -r requirements.txt
