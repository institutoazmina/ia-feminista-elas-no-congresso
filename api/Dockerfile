FROM python:3.13

WORKDIR /app

COPY ./api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY pre_processamento/ ./pre_processamento/

COPY ./api .

CMD ["python", "api.py"]