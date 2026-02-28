# NextStep — Student Risk Advisor

> Plataforma de inteligência preditiva para coordenadores pedagógicos identificarem e apoiarem estudantes em risco de defasagem acadêmica.

**Stack**: LSTM (PyTorch) · Flask 3 · React 18 · MLflow 2 · Groq LLM · Docker Compose  
**Dataset**: PEDE 2022-2024 (FIAP Datathon) · 1 156 alunos ativos · previsão para ciclo 2025  
**Último modelo**: v7 @prod · AUC=0.80 · F1=0.654 · threshold=0.314 (PR curve)

---

## Início Rápido

### Pré-requisitos

| Ferramenta | Versão mínima |
|---|---|
| Docker + Docker Compose | 24+ |

### 1. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edite .env e preencha GROQ_API_KEY=gsk_...
```

### 2. Subir o MLflow

```bash
docker compose up mlflow -d
# Aguarde ficar healthy antes de continuar
docker compose ps mlflow   # Status deve ser "healthy"
# MLflow UI: http://localhost:5000
```

### 3. Adicionar o dataset

Coloque o arquivo XLSX do PEDE na pasta git-ignorada:

```bash
mkdir -p backend/data/raw
cp /caminho/para/"BASE DE DADOS PEDE 2022-2024 - DATATHON.xlsx" backend/data/raw/
```

### 4. ETL — processar o dataset

```bash
docker compose run --rm --no-deps api python src/data_loader.py
# Gera em backend/data/processed/:
#   X_train.npy (600, 8)  y_train.npy (600,)
#   X_test.npy  (690, 8)  y_test.npy  (690,)
#   X_inference.npy (1156, 8)
#   scaler.pkl (RobustScaler fitado no treino)
#   students_meta.pkl (1156 registros)
```

### 5. Treinar o modelo

```bash
docker compose run --rm api python src/train_lstm.py
# - Split temporal: 2022→2023 treino | 2023→2024 teste
# - Validation set (20% do treino) para otimizar threshold via curva PR
# - Registra nextstep-lstm @staging no MLflow
```

### 6. Promover o modelo para produção

```bash
docker compose run --rm --no-deps api python -c "
import mlflow, os
from mlflow import MlflowClient
uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
mlflow.set_tracking_uri(uri)
c = MlflowClient(uri)
versions = c.search_model_versions(\"name='nextstep-lstm'\")
latest = max(int(v.version) for v in versions)
c.set_registered_model_alias('nextstep-lstm', 'prod', str(latest))
print(f'@prod definido na versao {latest}')
"
```

Ou manualmente: http://localhost:5000 → Models → `nextstep-lstm` → versão mais recente → Aliases → `prod`

### 7. Subir todos os serviços

```bash
docker compose up --build
```

| Serviço | URL |
|---|---|
| Frontend (React) | http://localhost:3000 |
| API (Flask) | http://localhost:8080 |
| MLflow UI | http://localhost:5000 |

> **Novo modelo em produção**: após treinar e promover o alias, basta `docker compose restart api`.

> **Dados persistidos**: `backend/data/processed/` e `mlruns/` são bind mounts — `docker compose down` **não** apaga os artefatos.

---

## Estrutura do Projeto

```
nextstep/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Flask factory (create_app)
│   │   ├── routes.py            # Endpoints REST
│   │   ├── prediction_cache.py  # Cache em memória (LSTM inference)
│   │   └── llm_service.py       # Integração Groq (LGPD-safe)
│   ├── src/
│   │   ├── data_loader.py       # ETL: PEDE XLSX → tensores (RobustScaler)
│   │   └── train_lstm.py        # Treinamento LSTM + MLflow + threshold PR
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_data_loader.py
│   │   ├── test_llm_service.py
│   │   └── test_model.py
│   ├── data/
│   │   ├── raw/                 # XLSX original (git-ignored)
│   │   └── processed/           # X_train/test/inference.npy, scaler.pkl (git-ignored)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── RiskBadge.tsx
│   │   │   ├── StudentListItem.tsx
│   │   │   ├── IndicatorCard.tsx
│   │   │   ├── AdvicePanel.tsx
│   │   │   └── ErrorState.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   └── StudentProfile.tsx
│   │   ├── services/api.ts
│   │   ├── types/student.ts
│   │   └── main.tsx
│   ├── Dockerfile
│   ├── nginx.conf
│   └── package.json
├── k8s/                         # Manifests Kubernetes (GKE)
├── .github/workflows/
│   ├── ci.yaml                  # CI: lint + testes
│   ├── train.yaml               # Treinamento manual
│   └── deploy.yaml              # Deploy GKE
├── docker-compose.yml
└── .env.example
```

---

## API

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "student_count": 1156 }
```

### `GET /api/students`
Lista estudantes ordenados por risco (desc).
```json
{
  "students": [
    { "student_id": 215, "display_name": "Aluno-750", "phase": "1B",
      "risk_score": 0.9877, "risk_tier": "high" }
  ],
  "total": 1156
}
```

### `GET /api/students/:id`
Perfil completo com indicadores. IPP é exibido mas **não** entra no modelo.
```json
{
  "student_id": 215, "display_name": "Aluno-750", "phase": "1B",
  "class_group": "A", "risk_score": 0.9877, "risk_tier": "high", "fase_num": 1,
  "indicators": { "iaa": 6.249, "ieg": 5.939, "ips": 4.38,
                  "ida": 3.75, "ipv": 3.177, "ipp": 4.063, "inde": 4.542,
                  "defasagem": -2 }
}
```

### `GET /api/students/:id/advice`
Sugestão pedagógica gerada pelo Groq (sempre HTTP 200).
```json
{
  "student_id": 215, "advice": "...", "is_fallback": false,
  "generated_at": "2026-02-28T12:00:00+00:00"
}
```

---

## Pipeline de ML

| Etapa | Detalhe |
|-------|---------|
| **Features** | IAA, IEG, IPS, IDA, IPV, INDE, defasagem, fase\_num (INPUT\_SIZE=8) |
| **IPP** | Display-only — ausente em 2022, imputado para exibição, não entra no modelo |
| **IAN** | Removido — data leakage (correlação 0.84–0.87 com o target) |
| **Split** | Temporal: 2022→2023 treino / 2023→2024 teste / 2024 inferência |
| **Missing (treino)** | DROP — linhas com null em qualquer feature são descartadas |
| **Missing (inferência)** | IMPUTE com medianas do treino (não pode perder alunos matriculados) |
| **Scaler** | `RobustScaler` (mediana+IQR, clip±5) — robusto a outliers |
| **Threshold** | Otimizado via curva PR no validation set (20% do treino) — nunca no test |
| **Modelo** | LSTM 1 camada hidden\_size=64, BCEWithLogitsLoss com pos\_weight |
| **Tracking** | MLflow: params, métricas, scaler como artefato, alias @staging/@prod |

---

## Testes

```bash
# Backend
cd backend
pip install -r requirements.txt
pytest tests/ -v

# Frontend
cd frontend
npm ci
npm test -- --run
```

---

## Qualidade de Código

```bash
# Python (Ruff, line-length=120)
ruff check backend/

# TypeScript
cd frontend && npm run lint
```

---

## Variáveis de Ambiente

| Variável | Descrição | Obrigatório |
|---|---|---|
| `GROQ_API_KEY` | Chave da API Groq | Sim (para advice) |
| `MLFLOW_TRACKING_URI` | URL do servidor MLflow | Sim |
| `VITE_API_BASE_URL` | URL base da API (frontend) | Sim |

---

## Thresholds de Risco

| Tier | Faixa | Badge |
|---|---|---|
| `high` | score ≥ 0.7 | 🔴 Red |
| `medium` | 0.3 ≤ score < 0.7 | 🟡 Yellow |
| `low` | score < 0.3 | 🟢 Green |

---

## Licença

Projeto acadêmico — FIAP Datathon 2026.
