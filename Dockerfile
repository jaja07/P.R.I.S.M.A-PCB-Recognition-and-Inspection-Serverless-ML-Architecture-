# ==========================================
# ÉTAPE 1 : BUILDER (Préparation)
# ==========================================
# On passe sur la version 3.13 demandée par ton pyproject.toml
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

FROM python:3.13-slim AS runner

WORKDIR /app

COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

COPY ./api ./api
COPY ./model/pcb_classifier_v1.0.0.onnx* ./model/
COPY ./model/data/PCB_DATASET/PCB_USED ./model/data/PCB_DATASET/PCB_USED

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]