FROM ubuntu:18.04
RUN apt update && apt upgrade -y
RUN apt install python3.7 python3-pip libsndfile1 -y
RUN python3.7 -m pip install -U pip
RUN pip3 install -U pip
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
RUN mkdir /models
RUN export LC_ALL=C.UTF-8 && export LANG=C.UTF-8
COPY src /src
WORKDIR /src
CMD python3.7 server.py