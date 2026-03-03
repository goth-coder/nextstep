# NextStep — Roadmap de Implementação

**GCloud Project**: `nextstep-fiap-project`
**Repositório**: https://github.com/goth-coder/nextstep.git
**Registry**: `us-central1-docker.pkg.dev/nextstep-fiap-project/nextstep/nextstep-{api,web}:<sha>`
**Cluster GKE**: `us-central1` (a criar)

---

## FASE 1 — ML Recheck: Target Engineering

> **Status**: ✅ Concluído — modelo retreinado com target corrigido, HPO regularizado.
>
> **Resultado final** (2026-03-01):
> - Val AUC-ROC: **0.7931** ✅ (gate ≥ 0.70)
> - Val F1 (oracle): **0.5872** ✅ (gate ≥ 0.55)
> - Val F1 (deployed): **0.5000** ⚠️ — threshold transfer degradation
> - Decision threshold: **0.0325** — modelo mal calibrado (ver diagnóstico)
> - Distribuição 2024: 36.9% alto, 5.2% médio, 57.9% baixo
>
> **Diagnóstico completo**: `docs/model-diagnostic-2026-03-01.md`
>
> **Conclusão**: AUC 0.79 válido para ranking. Classificação por tier com artefatos de calibração.
> Próxima melhoria: Platt Scaling + threshold operacional por percentil.

- [X] Abrir `backend/ml/data_loader.py` e auditar como `y` é construído
- [X] Confirmar se o target é Δdefasagem (t→t+1) ou estado pontual
- [X] **BUG ENCONTRADO E CORRIGIDO**: target era `defasagem_next < 0` (estado), corrigido para `defasagem_next < defasagem_current` (transição/piora)
- [X] Retreinar com novo target — AUC 0.7931, modelo mais realista
- [ ] Calibração de probabilidades (Platt Scaling) — thresholds operacionais
- [ ] Threshold por percentil operacional (máx 20% alto risco)
- [ ] Ajustar texto da UI: "probabilidade de piorar" em vez de "em risco agora"
- [ ] Benchmark XGBoost/LightGBM vs LSTM para dataset pequeno (1.156 registros)
- [X] Investigar gap Train F1 × Validation F1
  - [X] Split já é temporal por ano — correto, sem data leakage
  - [X] **BUG CORRIGIDO no HPO**: adicionado `weight_decay` ao Adam + search space Optuna
  - [X] WeightedRandomSampler + `pos_weight_multiplier` (0.5–4.0×)
  - [X] Oracle quality gate: gate em `test_f1_oracle` (melhor F1 possível no test set)

---

## FASE 2 — Kubernetes Completo

> **Status**: Manifests ✅ concluídos · Cluster GKE ⏳ a criar agora
>
> **Sizing estimado** (GKE `us-central1`, node pool `e2-standard-2` × 2 nós, ~$95/mês; spot ~$30/mês):
>
> | Workload | CPU req | Mem req | CPU lim | Mem lim | Réplicas |
> |---|---|---|---|---|---|
> | `nextstep-api` | 500m | 1Gi | 1000m | 2Gi | 2 |
> | `nextstep-web` | 50m | 64Mi | 100m | 128Mi | 2 |
> | `mlflow` | 250m | 512Mi | 500m | 1Gi | 1 |

### 2.1 Estrutura de manifests

- [X] Criar `k8s/namespace.yaml`
- [X] Atualizar `k8s/backend-deployment.yaml` + `securityContext` non-root
- [X] Atualizar `k8s/frontend-deployment.yaml` + `securityContext` non-root
- [X] Criar `k8s/mlflow-deployment.yaml` + `k8s/mlflow-service.yaml`
- [X] Criar `k8s/mlflow-pvc.yaml`
- [X] Criar `k8s/hpa.yaml` (min=2 max=5, CPU 70%)
- [X] Criar `k8s/network-policy.yaml`
- [X] `namespace: nextstep` em todos os manifests
- [X] Domínio atualizado para `nextstep-advisor.uk` em todos os manifests

### 2.2 Cluster GKE

- [ ] Criar cluster GKE:
  ```bash
  gcloud container clusters create nextstep \
    --project nextstep-fiap-project \
    --region us-central1 \
    --num-nodes 2 \
    --machine-type e2-standard-2 \
    --enable-autoscaling --min-nodes 2 --max-nodes 4
  ```
- [ ] Obter credenciais do cluster:
  ```bash
  gcloud container clusters get-credentials nextstep \
    --region us-central1 --project nextstep-fiap-project
  ```
- [ ] Criar namespace + secret:
  ```bash
  kubectl apply -f k8s/namespace.yaml
  kubectl create secret generic nextstep-secrets \
    --from-literal=groq-api-key=$GROQ_API_KEY -n nextstep
  ```
- [ ] Aplicar todos os manifests:
  ```bash
  kubectl apply -f k8s/
  ```
- [ ] Verificar rollout:
  ```bash
  kubectl rollout status deployment/nextstep-api -n nextstep
  kubectl rollout status deployment/nextstep-web -n nextstep
  kubectl get pods -n nextstep
  ```

---

## FASE 3 — HTTPS + DNS

> **Status**: Manifests criados ✅ · DNS comprado ✅ · Configuração pendente após cluster
>
> **Domínio**: `nextstep-advisor.uk` (Cloudflare)  
> **Estratégia**: NGINX Ingress Controller + cert-manager + Let's Encrypt

### 3.1 DNS

- [X] Domínio adquirido: `nextstep-advisor.uk` via Cloudflare
- [ ] Após cluster criado, obter IP do Ingress: `kubectl get ingress -n nextstep`
- [ ] Na Cloudflare: criar registro A `@ → <IP externo>`, proxy **DNS Only** (cinza) durante emissão do cert
- [ ] Aguardar propagação DNS (até 48h, geralmente minutos via Cloudflare)
- [ ] Após cert emitido, ligar proxy laranja (Cloudflare CDN + DDoS)

### 3.2 Ingress + TLS

- [ ] Instalar NGINX Ingress Controller:
  ```bash
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
  helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace
  ```
- [ ] Instalar cert-manager:
  ```bash
  helm repo add jetstack https://charts.jetstack.io
  helm install cert-manager jetstack/cert-manager \
    --namespace cert-manager --create-namespace --set installCRDs=true
  ```
- [X] `k8s/cluster-issuer.yaml` criado (Let's Encrypt staging + prod) com `admin@nextstep-advisor.uk`
- [X] `k8s/ingress.yaml` criado com TLS, host `nextstep-advisor.uk`
- [ ] Aplicar issuer + ingress: `kubectl apply -f k8s/cluster-issuer.yaml && kubectl apply -f k8s/ingress.yaml`
- [ ] Verificar certificado: `kubectl describe certificate -n nextstep`
- [ ] Testar HTTPS: `curl -I https://nextstep-advisor.uk/health`

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
- [ ] Adicionar step de scan de vulnerabilidades: `gcloud artifacts docker images scan us-central1-docker.pkg.dev/nextstep-fiap-project/nextstep/nextstep-api:$SHA`
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
gcloud artifacts docker images list us-central1-docker.pkg.dev/nextstep-fiap-project/nextstep

# Logs da API em produção
kubectl logs -l app=nextstep-api -n nextstep --tail=100 -f

# Escalar manualmente
kubectl scale deployment nextstep-api --replicas=3 -n nextstep

# Aplicar um manifest específico
kubectl apply -f k8s/ingress.yaml -n nextstep
```
