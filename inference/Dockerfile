FROM python:3.8

COPY . /src

COPY ./requirements.txt /src/requirements.txt

WORKDIR src

EXPOSE 8051:8051

RUN pip install -r requirements.txt
    
CMD [ "opyrator",  "launch-ui", "app:predict", "--port", "8051" ]