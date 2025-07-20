FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치 (레이어 분리)
RUN apt-get update && apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python 의존성 먼저 설치 (캐싱 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (마지막에 복사하여 코드 변경 시 캐싱 활용)
COPY main.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
