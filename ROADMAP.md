# NextStep — Roadmap de Implementação

**GCloud Project**: `nextstep-fiap-project`
**Repositório**: https://github.com/goth-coder/nextstep.git
**Registry**: `gcr.io/nextstep-fiap-project/nextstep-{api,web}:<sha>`
**Cluster GKE**: `us-central1` (a criar)

---

## FASE 1 — ML Recheck: Target Engineering

> **Contexto**: O modelo atual pode estar prevendo o *estado atual* de risco em vez da *transição futura*. A proposta de valor do NextStep é: dado o estado de hoje, qual a probabilidade de o aluno *piorar* no próximo ciclo?
>
> **Caso real**: Aluno 1028, defasagem = −2 (adiantado), índices bons → score de risco baixo. Correto? Depende de como o `y` foi construído.
>
> **Target correto**: `y = 1 se defasagem(2024) > defasagem(2023)` (aluno regrediu no ciclo) -  é assim q ta sendo feito a rede?
> **Target suspeito**: `y = 1 se defasagem(2024) > 0` (estado atual, não transição)
>
> **Gap treino × validação (v37)**:
> - Train F1 ≈ **0.87** · Validation F1 ≈ **0.76** → delta de ~0.11
> - Causa confirmada: **shift de distribuição do target** entre os dois ciclos — train: 61% positivo (piorou 22→23), test: 44% positivo (piorou 23→24). O modelo treina com base numa taxa de piora muito maior do que a que encontra no test. Isso naturalmente eleva o recall e F1 no treino e penaliza na validação.
> - Causa secundária: **bug no HPO** — modelos `num_layers=1` (maioria dos trials) tinham `dropout=0.0` E sem `weight_decay` → zero regularização. Corrigido.
> - O split temporal (2022→2023 como train, 2023→2024 como test) está **correto** — é holdout por ciclo, sem leakage.

- [X] Abrir `backend/ml/data_loader.py` e auditar como `y` é construído
- [X] Confirmar se o target é Δdefasagem (t→t+1) ou estado pontual
- [X] **BUG ENCONTRADO E CORRIGIDO**: target era `defasagem_next < 0` (estado), corrigido para `defasagem_next < defasagem_current` (transição/piora)
- [ ] Retreinar com novo target e comparar AUC / F1 / threshold com v37
- [ ] Ajustar texto da UI: "probabilidade de piorar" em vez de "em risco agora"
- [ ] Validar aluno 1028 manualmente após retreino (score deve continuar baixo se índices bons)
- [ ] Investigar gap Train F1 ≈ 0.87 × Validation F1 ≈ 0.76 (delta ~0.11)
  - [X] ~~Substituir split aleatório~~ — split já é temporal por ano (2022→train, 2023→test), não havia esse problema
  - [X] **BUG CORRIGIDO no HPO**: modelos com `num_layers=1` tinham `dropout=0.0` forçado E sem `weight_decay` → zero regularização. Corrigido: adicionado `weight_decay` (L2) ao search space do Optuna (`1e-5`..`1e-2`, log scale) e ao `TrainConfig`/`Adam` optimizer
  - [ ] Retreinar HPO (tune.py --trials 30) com novo search space e avaliar se gap diminui
  - [ ] Verificar se há data leakage no feature engineering (features calculadas com dados futuros)

---

## FASE 2 — Kubernetes Completo

> **Sizing estimado** (GKE `us-central1`, node pool `e2-standard-2` × 2 nós, ~$95/mês; spot ~$30/mês):
>
> | Workload | CPU req | Mem req | CPU lim | Mem lim | Réplicas |
> |---|---|---|---|---|---|
> | `nextstep-api` | 500m | 1Gi | 1000m | 2Gi | 2 |
> | `nextstep-web` | 50m | 64Mi | 100m | 128Mi | 2 |
> | `mlflow` | 250m | 512Mi | 500m | 1Gi | 1 |
>
> **Nota sobre registry**: Os manifests já usam `gcr.io/PROJECT_ID/...` — correto. O Artifact Registry (`pkg.dev`) é o successor; GCR redireciona automaticamente. Nenhuma mudança de URI necessária.

### 2.1 Estrutura de manifests

- [X] Criar `k8s/namespace.yaml` (namespace `nextstep`)
- [X] Atualizar `k8s/backend-deployment.yaml`: resources conforme tabela acima + `securityContext` non-root
- [X] Atualizar `k8s/frontend-deployment.yaml`: resources conforme tabela acima + `securityContext` non-root
- [X] Criar `k8s/mlflow-deployment.yaml` + `k8s/mlflow-service.yaml`
- [X] Criar `k8s/mlflow-pvc.yaml` (PersistentVolumeClaim para dados MLflow)
- [X] Criar `k8s/hpa.yaml` (HorizontalPodAutoscaler para `nextstep-api`, min=2 max=5, CPU 70%)
- [X] Criar `k8s/network-policy.yaml` (API só aceita tráfego do web + ingress)
- [X] Adicionar `namespace: nextstep` em todos os manifests

### 2.2 Cluster GKE

- [ ] Criar cluster: `gcloud container clusters create nextstep --project nextstep-fiap-project --region us-central1 --num-nodes 2 --machine-type e2-standard-2 --enable-autoscaling --min-nodes 2 --max-nodes 4`
- [ ] Criar Service Account com roles: `roles/storage.objectViewer`, `roles/artifactregistry.reader`
- [ ] Criar secret `nextstep-secrets` no cluster: `kubectl create secret generic nextstep-secrets --from-literal=groq-api-key=$GROQ_API_KEY -n nextstep`
- [ ] Aplicar manifests: `kubectl apply -f k8s/`
- [X] Verificar rollout: adicionado ao deploy.yaml como step automático (`kubectl rollout status --timeout=120s`)

---

## FASE 3 — HTTPS + DNS

> **Opção A (recomendada)**: NGINX Ingress Controller + cert-manager + Let's Encrypt → mais controle, funciona com qualquer domínio
> **Opção B (mais simples)**: GCE Ingress com Google-managed SSL → zero config de cert, mas preso ao GCP

### 3.1 DNS

- [ ] Adquirir/configurar domínio (ex: `nextstep.app` ou subdomínio)
- [ ] Após deploy, obter IP externo do Ingress: `kubectl get ingress -n nextstep`
- [ ] Criar registro A: `nextstep.app → <IP externo>` no registrador de domínio
- [ ] Aguardar propagação DNS (TTL, pode levar até 48h)

### 3.2 Ingress + TLS

- [ ] Instalar NGINX Ingress Controller: `helm install ingress-nginx ingress-nginx/ingress-nginx`
- [ ] Instalar cert-manager: `helm install cert-manager jetstack/cert-manager --set installCRDs=true`
- [X] Criar `k8s/cluster-issuer.yaml` (Let's Encrypt prod + staging)
- [X] Criar `k8s/ingress.yaml` com TLS, rotas `/api/*` → `nextstep-api-service:8080` e `/` → `nextstep-web-service:80`
- [ ] Verificar certificado: `kubectl describe certificate -n nextstep`
- [ ] Testar HTTPS: `curl -I https://nextstep.app/health`

---

## FASE 4 — CI/CD Hardening

> **Lacunas atuais**:
> - `deploy.yaml` não valida rollout nem faz smoke test pós-deploy
> - `ci.yaml` não roda no push para `main`
> - Sem rollback automático em falha de deploy

### 4.1 deploy.yaml — pós-deploy

- [X] Adicionar `kubectl rollout status deployment/nextstep-api --timeout=120s -n nextstep`
- [X] Adicionar `kubectl rollout status deployment/nextstep-web --timeout=60s -n nextstep`
- [X] Adicionar smoke test: `curl -f https://nextstep.app/health` (falha se não 200)
- [X] Adicionar smoke test: `curl -f https://nextstep.app/api/students?limit=1`
- [X] Adicionar step de rollback automático em falha: `kubectl rollout undo deployment/nextstep-api -n nextstep`

### 4.2 ci.yaml — cobertura de branches

- [ ] Adicionar `main` nos branches de push do `ci.yaml` (ou garantir via PR → CI roda antes do merge)
- [ ] Habilitar **Container Analysis** no GCR: `gcloud services enable containeranalysis.googleapis.com --project nextstep-fiap-project`
- [ ] Adicionar step de scan de vulnerabilidades: `gcloud artifacts docker images scan gcr.io/nextstep-fiap-project/nextstep-api:$SHA`
- [X] Configurar GitHub Secrets necessários:
  - `GCP_PROJECT_ID` = `nextstep-fiap-project`
  - `GCP_SA_KEY` (JSON da service account)
  - `GKE_CLUSTER_NAME`
  - `GROQ_API_KEY`

---

## FASE 5 — Segurança da API

> **Checklist de segurança**:
>
> | Item | Status |
> |---|---|
> | `GROQ_API_KEY` em K8s Secret | ✅ |
> | HTTPS/TLS | ❌ → Fase 3 |
> | NetworkPolicy | ❌ → Fase 2 |
> | Non-root containers | ❌ → Fase 2 |
> | Image scanning | ❌ → Fase 4 |
> | Rate limiting na API | ❌ → Fase 5 |
> | CORS restrito ao domínio | verificar |

- [X] Instalar `Flask-Limiter` em `backend/requirements.txt`
- [X] Configurar rate limiting: `20/hour` por IP no endpoint de IA (`/api/students/<id>/advice`), `200/hour` + `30/min` default global
- [X] Restringir CORS em produção: lê `ALLOWED_ORIGIN` env var (ConfigMap seta `https://nextstep.app`, fallback `*` local)
- [X] Adicionar `securityContext` nos deployments K8s: `runAsNonRoot: true`, `readOnlyRootFilesystem: true`
- [ ] Auditar `backend/` e `frontend/` por secrets hardcoded: `grep -r "gsk_\|sk-\|AIza" .`
- [X] Adicionar header `X-Content-Type-Options: nosniff` e `X-Frame-Options: DENY` no nginx do frontend

---

## Ordem de Execução Recomendada

```
HOJE     → FASE 1: ML recheck (auditar target, retreinar se necessário)
SEMANA 1 → FASE 2: K8s manifests completos + criar cluster GKE
SEMANA 1 → FASE 3: Ingress + HTTPS + DNS
SEMANA 2 → FASE 4: CI/CD hardening (smoke tests, rollback, scanning)
SEMANA 2 → FASE 5: Segurança API (rate limit, CORS, securityContext)
```

---

## Referências Rápidas

```bash
# Autenticar no GKE
gcloud container clusters get-credentials <CLUSTER_NAME> \
  --region us-central1 --project nextstep-fiap-project

# Ver imagens no GCR
gcloud container images list --repository=gcr.io/nextstep-fiap-project

# Logs da API em produção
kubectl logs -l app=nextstep-api -n nextstep --tail=100 -f

# Escalar manualmente
kubectl scale deployment nextstep-api --replicas=3 -n nextstep

# Aplicar um manifest específico
kubectl apply -f k8s/ingress.yaml -n nextstep
```
