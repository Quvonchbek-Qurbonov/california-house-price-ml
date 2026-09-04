FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip \
    && mkdir -p /app/app/dataset \
    && curl -L -o /tmp/california-housing-prices.zip \
       https://www.kaggle.com/api/v1/datasets/download/camnugent/california-housing-prices \
    && unzip /tmp/california-housing-prices.zip -d /app/app/dataset \
    && rm /tmp/california-housing-prices.zip \
    && rm -rf /var/lib/apt/lists/*


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]