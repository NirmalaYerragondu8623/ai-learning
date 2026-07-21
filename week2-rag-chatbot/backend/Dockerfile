FROM python:3.12-slim
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]