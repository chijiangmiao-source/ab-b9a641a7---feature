# Deep-sea buoy reachability-audit service.
# Exact rational solver (fractions.Fraction + two-phase simplex); stdlib only,
# so the image builds with no package downloads beyond the base image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /srv

COPY app ./app
COPY tests ./tests
COPY verify ./verify

EXPOSE 8080

# Liveness probe used by Compose (depends_on: service_healthy) and Docker.
HEALTHCHECK --interval=5s --timeout=3s --start-period=5s --retries=12 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2).status==200 else 1)"

CMD ["python", "-m", "app.server"]
