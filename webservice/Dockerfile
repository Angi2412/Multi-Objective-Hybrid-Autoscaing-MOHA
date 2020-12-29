FROM python:3.8-slim
RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app
RUN pip3 install -r requirements.txt
ADD . /app
COPY test_webservice.py /app
EXPOSE 5000
ENTRYPOINT ["python", "test_webservice.py"]