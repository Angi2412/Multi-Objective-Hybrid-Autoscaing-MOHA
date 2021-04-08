FROM python:3.8

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
COPY prod.env .env
CMD ["autoscaler.py"]
ENTRYPOINT ["python3"]