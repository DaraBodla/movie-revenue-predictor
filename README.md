# 🎬 Movie Revenue Predictor

Not a notebook. A full system with a REST API, orchestrated retraining pipelines, CI/CD, and a live dashboard. Built solo.

---

## what it does

Takes movie metadata and returns a revenue prediction. The model layer is a stacked ensemble of gradient boosting models. FastAPI handles serving. Prefect manages retraining. Streamlit sits on top for exploration and live predictions.

---

## architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard                  │
│              (exploration + live predictions)            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Backend                        │
│              /predict  /retrain  /metrics                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Ensemble Model Layer                         │
│     XGBoost  ·  LightGBM  ·  CatBoost  →  Meta-learner  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│             Prefect Pipeline Orchestration                │
│       ingest → preprocess → train → evaluate → save      │
└─────────────────────────────────────────────────────────┘
```

---

## stack

| Layer | Tools |
|---|---|
| Models | XGBoost · LightGBM · CatBoost |
| API | FastAPI · Uvicorn · Pydantic |
| Orchestration | Prefect |
| Dashboard | Streamlit |
| Containers | Docker · Docker Compose |
| CI/CD | GitHub Actions |
| Data | Pandas · NumPy · scikit-learn |

---

## project structure

```
movie-revenue-predictor/
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py           # request/response models
│   └── routes/
│       ├── predict.py
│       └── metrics.py
├── pipeline/
│   ├── ingest.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── ensemble.py
├── dashboard/
│   └── app.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## quickstart

**Docker**

```bash
git clone https://github.com/DaraBodla/movie-revenue-predictor.git
cd movie-revenue-predictor
docker-compose up --build
```

API at `http://localhost:8000`
Dashboard at `http://localhost:8501`

**Local**

```bash
pip install -r requirements.txt

uvicorn api.main:app --reload

streamlit run dashboard/app.py

python -m pipeline.train
```

---

## API

```bash
POST /predict
```

```json
{
  "budget": 150000000,
  "genres": ["Action", "Adventure"],
  "runtime": 132,
  "release_month": 6,
  "cast_popularity": 87.4,
  "director_experience": 12
}
```

```json
{
  "predicted_revenue": 412800000,
  "confidence_interval": [318000000, 507600000],
  "model_version": "v1.3.2"
}
```

---

## CI/CD

Every push to `main` runs lint, tests, builds the Docker image, deploys, and hits a smoke test. Retraining is on a Prefect schedule. A new model only gets promoted if it beats the current one on held-out eval.

---

## results

| Metric | Score |
|---|---|
| R² | 0.87 |
| MAE | ~$18M |
| RMSE | ~$34M |

The ensemble beats any single model by around 6 to 9 percent on R².

---

## built by

[Dara Bodla](https://github.com/DaraBodla)
