# VisionGuard (23L-0951_A2_Ops)

![CI](https://github.com/bhatti-zayan/23L-0951_A2_Ops/actions/workflows/ci.yml/badge.svg)

Classifies an uploaded image as **AI-generated** or **real**: a MobileNetV2 model fine-tuned with
TensorFlow/Keras, tracked with MLflow on DagsHub, served by Flask, logged to PostgreSQL and run with Docker Compose.
MLOps Assignment 2, FAST NUCES Lahore.

| Part | Where |
|---|---|
| Data + model versioning | DVC, DagsHub storage (`data/raw.dvc`, `models/champion.keras.dvc`) |
| Experiments + registry | MLflow on DagsHub, registered model `23L-0951_A2_Ops_visionguard` (alias `champion`) |
| Training notebooks | `notebooks/` (Google Colab) |
| Web app | `app/` (Flask), shared preprocessing in `src/data.py` |
| Containers | `Dockerfile`, `docker-compose.yml` |
| CI | `.github/workflows/ci.yml` (flake8, pytest, docker build) |

## Run locally with Docker
```bash
cp .env.example .env                 # then set real secrets
dvc pull models/champion.keras       # fetch the champion model once
docker compose up -d --build
# open http://localhost:8000
```
