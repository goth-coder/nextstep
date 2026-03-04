FROM ghcr.io/mlflow/mlflow:v2.14.3

RUN pip install "google-cloud-storage>=2.10,<3" "protobuf>=3.20,<4" --quiet

COPY sync.py /sync.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PORT=8080
ENV MLFLOW_BUCKET=""
ENV SYNC_INTERVAL=300

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
