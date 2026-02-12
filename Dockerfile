FROM python:3.12-rc-slim-buster
WORKDIR /app
COPY . /app
RUN apt update && apt install -y build-essential

RUN pip install -r requirements.txt
CMD [ "python", "app.py" ]