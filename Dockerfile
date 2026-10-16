# ==============================================================================
# S1stem v1.3.0 Standalone Container
# Fast System 1 Contrastive Decision Engine (CLM) Microservice
# ==============================================================================

FROM node:20-slim

WORKDIR /app

# Install curl for healthcheck & CA certificates for secure runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for optimal Docker layer caching
COPY package.json package-lock.json* ./

# Install production dependencies only
RUN npm ci --omit=dev || npm install --omit=dev

# Copy core engine, microservice server, and scripts
COPY core/ ./core/
COPY server/ ./server/
COPY scripts/ ./scripts/

# Pre-bake ONNX model into container image for zero-download instant startup
ARG PREBAKE_MODEL=true
RUN if [ "$PREBAKE_MODEL" = "true" ] ; then \
      node scripts/download-model.js --dest ./models/s1stem-bge-large ; \
    fi

ENV PORT=8000
ENV HOST=0.0.0.0
ENV S1STEM_MODEL_PATH=/app/models/s1stem-bge-large
ENV NODE_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["node", "server/s1stem-server.js"]
