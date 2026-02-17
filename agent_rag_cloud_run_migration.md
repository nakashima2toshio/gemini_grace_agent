# Gemini Agent RAG → Cloud Run 移行戦略

5 フェーズ · 19 タスク · Streamlit + Qdrant + Redis + Gemini API

---

## アーキテクチャ比較

### 現在の構成（ローカル / GCP VM）

| コンポーネント | 説明 |
|:---|:---|
| 💻 MacBook Air M2 | 開発環境 |
| 🖥️ Streamlit | port 8501 |
| 🔍 Qdrant | Docker / port 6333 |
| 📦 Redis | Docker / port 6379 |
| ⚙️ Celery Worker | ローカルプロセス |
| 🤖 Gemini API | 外部API |

### Cloud Run 移行後の構成

| コンポーネント | 説明 |
|:---|:---|
| ☁️ Cloud Run | Streamlit App |
| 🔍 Qdrant Cloud | マネージドDB |
| 📦 Memorystore | Redis（任意） |
| 🔐 Secret Manager | APIキー管理 |
| 📦 Artifact Registry | Dockerイメージ |
| 🤖 Gemini API | 外部API |

---

## 💡 重要な設計判断

**Celeryの扱い：** Cloud RunではCeleryワーカー常駐が困難。2つの選択肢あり。

- **選択肢①** Memorystore + Cloud Run Jobs で置換
- **選択肢②** バッチ処理は既存GCE VMに残す

**推奨：** Streamlit UI のみ Cloud Run 化が最もシンプル。

---

## Phase 1: 準備・設計フェーズ（1〜2日）

アーキテクチャ決定 & GCPリソース準備

### 1-1. GCPプロジェクト・API有効化 【必須】

GCPコンソールで以下のAPIを有効化：

- Cloud Run API
- Artifact Registry API
- Secret Manager API
- Memorystore for Redis API（Celery使用時）
- VPC Access Connector API（内部通信用）

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  redis.googleapis.com \
  vpcaccess.googleapis.com
```

### 1-2. Artifact Registry リポジトリ作成 【必須】

Dockerイメージを格納するリポジトリを作成。

```bash
gcloud artifacts repositories create gemini-rag \
  --repository-format=docker \
  --location=asia-northeast1 \
  --description="Gemini Agent RAG images"
```

### 1-3. Secret Manager にAPIキー登録 【必須】

`.env` のAPIキーをSecret Managerで安全に管理。Cloud Runサービスから参照可能にする。

```bash
# Gemini API Key
echo -n "your_gemini_api_key" | \
  gcloud secrets create GEMINI_API_KEY --data-file=-

# Cohere API Key（Rerank使用時）
echo -n "your_cohere_api_key" | \
  gcloud secrets create COHERE_API_KEY --data-file=-
```

### 1-4. VPCネットワーク・サーバーレスVPCコネクタ作成 【必須】

Cloud Run → Qdrant/Redis 間の内部通信にVPCコネクタが必要。Memorystore（Redis）はVPC内でしかアクセスできない。

```bash
gcloud compute networks vpc-access connectors create rag-connector \
  --region=asia-northeast1 \
  --range=10.8.0.0/28
```

---

## Phase 2: インフラ構築フェーズ（1〜2日）

Qdrant・Redis のマネージド化

### 2-1. 【選択A】Qdrant Cloud（推奨）

Qdrant公式マネージドサービスを利用。運用負荷ゼロ、自動バックアップ、スケーリング対応。

Free Tier: 1GBストレージ、1ノード（開発には十分）

config.py の QdrantConfig を変更：

- HOST → Qdrant Cloud のエンドポイント
- API_KEY → Qdrant Cloud のAPIキー

```python
# config.py の変更例
class QdrantConfig:
    HOST = os.getenv("QDRANT_HOST", "your-cluster.cloud.qdrant.io")
    PORT = 6333
    API_KEY = os.getenv("QDRANT_API_KEY")  # 追加
    URL = f"https://{HOST}:{PORT}"
```

### 2-2. 【選択B】GCE上にQdrant VM（低コスト）

既存のGCP VMにQdrantを残す方法。コスト最小だが、可用性・運用は自己管理。

e2-small（2vCPU/2GB）で月額約$15〜20。Cloud Run → VPC経由でアクセス。

```yaml
# 既存VMのQdrantをそのまま利用
# docker-compose.yml は Qdrant のみに縮小
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped
```

### 2-3. Memorystore for Redis 作成 【必須（Celery使用時）】

Celeryのブローカー/バックエンドとして使用。BASIC tier（1GB）で月額約$35。

※ Celeryを使わない構成にする場合は不要（Phase 3で Cloud Run Jobs への置き換えを検討）

```bash
gcloud redis instances create rag-redis \
  --size=1 \
  --region=asia-northeast1 \
  --redis-version=redis_7_0 \
  --tier=basic

# 接続情報取得
gcloud redis instances describe rag-redis \
  --region=asia-northeast1 \
  --format="value(host,port)"
```

---

## Phase 3: アプリケーション改修フェーズ（2〜3日）

コンテナ化 & 環境変数の外部化

### 3-1. config.py を環境変数ベースに改修 【必須】

ハードコードされた localhost 参照を全て環境変数化。Cloud Run では Secret Manager + 環境変数で注入。

```python
# config.py 改修例
class QdrantConfig:
    HOST = os.getenv("QDRANT_HOST", "localhost")
    PORT = int(os.getenv("QDRANT_PORT", "6333"))
    API_KEY = os.getenv("QDRANT_API_KEY", None)

class CeleryConfig:
    BROKER_URL = os.getenv(
        "CELERY_BROKER_URL",
        "redis://localhost:6379/0"
    )
    RESULT_BACKEND = os.getenv(
        "CELERY_RESULT_BACKEND",
        "redis://localhost:6379/0"
    )
```

### 3-2. Dockerfile 作成 【必須】

Streamlitアプリ用のDockerfileを作成。Cloud Run は PORT 環境変数でリッスンポートを指定。マルチステージビルドで軽量化。

```dockerfile
# Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages \
     /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Cloud Run は PORT 環境変数を自動設定
ENV PORT=8501
EXPOSE 8501

# Streamlit設定
RUN mkdir -p ~/.streamlit && \
    echo '[server]\nheadless = true\nport = 8501\n\
enableCORS = false\nenableXsrfProtection = false\n\
[browser]\nserverAddress = "0.0.0.0"' \
    > ~/.streamlit/config.toml

CMD ["streamlit", "run", "agent_rag.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

### 3-3. .dockerignore 作成 【必須】

不要ファイルをビルドコンテキストから除外し、イメージを軽量化。

```text
# .dockerignore
.git
.gitignore
.idea
.venv
venv
__pycache__
*.pyc
.env
.DS_Store
OUTPUT/
temp_uploads/
logs/
*.md
tests/
docker-compose.yml
```

### 3-4. ローカルでDockerビルド＆テスト 【必須】

Cloud Runにデプロイ前にローカルでテスト。M2 Macの場合、`--platform linux/amd64` でクロスビルド。

```bash
# ビルド（Cloud Run用にamd64）
docker build --platform linux/amd64 \
  -t gemini-rag:test .

# ローカルテスト（Qdrant/Redisはdocker-composeで起動済み前提）
docker run --rm -p 8501:8501 \
  --network qdrant-network \
  -e GEMINI_API_KEY=your_key \
  -e QDRANT_HOST=qdrant \
  -e CELERY_BROKER_URL=redis://redis:6379/0 \
  gemini-rag:test
```

---

## Phase 4: デプロイフェーズ（1日）

Cloud Run へのデプロイ & 動作確認

### 4-1. Artifact Registry へイメージPush 【必須】

ビルドしたイメージをArtifact Registryに格納。

```bash
# 認証設定
gcloud auth configure-docker \
  asia-northeast1-docker.pkg.dev

# タグ付け & Push
docker tag gemini-rag:test \
  asia-northeast1-docker.pkg.dev/YOUR_PROJECT/gemini-rag/app:v1

docker push \
  asia-northeast1-docker.pkg.dev/YOUR_PROJECT/gemini-rag/app:v1
```

### 4-2. Cloud Build で自動ビルド（推奨） 【推奨】

手動Pushの代わりに、Cloud Buildでビルド→Push自動化。cloudbuild.yaml を作成。

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--platform=linux/amd64'
      - '-t'
      - 'asia-northeast1-docker.pkg.dev/$PROJECT_ID/gemini-rag/app:$SHORT_SHA'
      - '.'
images:
  - 'asia-northeast1-docker.pkg.dev/$PROJECT_ID/gemini-rag/app:$SHORT_SHA'
```

```bash
# ビルド実行
gcloud builds submit --config=cloudbuild.yaml .
```

### 4-3. Cloud Run サービスデプロイ 【必須】

Streamlitアプリを Cloud Run にデプロイ。WebSocket対応のため session-affinity と タイムアウトを設定。メモリは Embedding処理があるため 2Gi 以上推奨。

```bash
gcloud run deploy gemini-rag-app \
  --image=asia-northeast1-docker.pkg.dev/YOUR_PROJECT/gemini-rag/app:v1 \
  --region=asia-northeast1 \
  --platform=managed \
  --port=8501 \
  --memory=2Gi \
  --cpu=2 \
  --timeout=3600 \
  --session-affinity \
  --min-instances=0 \
  --max-instances=3 \
  --vpc-connector=rag-connector \
  --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest,\
GOOGLE_API_KEY=GEMINI_API_KEY:latest,\
COHERE_API_KEY=COHERE_API_KEY:latest,\
QDRANT_API_KEY=QDRANT_API_KEY:latest" \
  --set-env-vars="QDRANT_HOST=your-qdrant-host,\
CELERY_BROKER_URL=redis://REDIS_HOST:6379/0" \
  --allow-unauthenticated
```

### 4-4. 動作確認チェックリスト 【必須】

デプロイ後に以下を順番に確認：

- [ ] Cloud Run URLでStreamlit UIが表示される
- [ ] Qdrant接続：コレクション一覧が取得できる
- [ ] Qdrant検索：ベクトル検索が正常動作する
- [ ] Agent(ReAct)：質問→検索→回答の全フロー
- [ ] WebSocket：ページ遷移でセッションが切れない
- [ ] ログ：Cloud Logging でエラーが出ていない

```bash
# Cloud Run URL確認
gcloud run services describe gemini-rag-app \
  --region=asia-northeast1 \
  --format="value(status.url)"

# ログ確認
gcloud logging read \
  "resource.type=cloud_run_revision AND \
   resource.labels.service_name=gemini-rag-app" \
  --limit=50 --format=json
```

---

## Phase 5: 運用・最適化フェーズ（継続的）

監視・CI/CD・コスト最適化

### 5-1. カスタムドメイン設定 【任意】

独自ドメインをCloud Runサービスにマッピング。

```bash
gcloud run domain-mappings create \
  --service=gemini-rag-app \
  --domain=rag.your-domain.com \
  --region=asia-northeast1
```

### 5-2. Cloud Monitoring アラート設定 【推奨】

エラー率、レイテンシ、メモリ使用量の監視。Gemini APIのレート制限エラー（429）も監視対象に。

```text
# gcloud CLI or GCPコンソールで設定
# 主要メトリクス：
#  - request_count（status=5xx）
#  - request_latencies（p95 > 30s）
#  - container/memory/utilization（> 80%）
```

### 5-3. GitHub Actions CI/CD パイプライン 【推奨】

mainブランチへのPushで自動デプロイ。Workload Identity Federation で認証。

```yaml
# .github/workflows/deploy.yml
name: Deploy to Cloud Run
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.SA_EMAIL }}
      - uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: gemini-rag-app
          region: asia-northeast1
          source: .
```

### 5-4. コスト最適化 【推奨】

Cloud Run の課金はリクエスト処理中のみ。`min-instances=0` でアイドル時コストゼロ。

**月額コスト目安（軽量利用）：**

| サービス | 月額 |
|:---|:---|
| Cloud Run | $0〜5（従量課金、無料枠あり） |
| Qdrant Cloud Free | $0 |
| Memorystore Redis | ~$35（Celery使用時） |
| Artifact Registry | ~$1 |
| Secret Manager | ~$0.06 |

Celeryを使わない構成なら月額 **$5以下** も可能。

```bash
# 予算アラート設定
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="RAG App Budget" \
  --budget-amount=50USD \
  --threshold-rule=percent=80
```
