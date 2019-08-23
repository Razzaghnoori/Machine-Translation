FROM tensorflow/tensorflow:1.14.0-gpu-py3

WORKDIR /opt/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data data

COPY start.sh start.sh
COPY main.py main.py

CMD ./start.sh