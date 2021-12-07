FROM python:3.6.1

RUN apt-get update && \
    apt-get install -y \
    libblas-dev && \
    rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /home

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /home/requirements.txt

ADD . /code
WORKDIR /code

ENV PYTHONPATH=/code:$PYTHONPATH

CMD bash 
