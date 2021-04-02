FROM python:3.8

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["autoscaler.py"]
ENTRYPOINT ["python3"]
EXPOSE 5000