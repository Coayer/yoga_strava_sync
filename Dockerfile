FROM python:3.13.1-slim-bookworm
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["gunicorn", "--threads", "1", "--bind", "0.0.0.0:80", "--timeout", "120", "wsgi:app"]
