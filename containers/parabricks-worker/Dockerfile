# syntax=docker/dockerfile:1.7

# The base is both version- and manifest-digest-pinned.  Per-job biological
# identities enter only through the read-only request file passed to ENTRYPOINT.
FROM nvcr.io/nvidia/clara/clara-parabricks:4.7.0-1@sha256:28f2191158fd14421958fc275e219f22f6caaa3755ae3250c0f04883a880c212
ARG WORKER_SHA256

LABEL org.opencontainers.image.base.name="nvcr.io/nvidia/clara/clara-parabricks:4.7.0-1@sha256:28f2191158fd14421958fc275e219f22f6caaa3755ae3250c0f04883a880c212" \
      io.bio-neb.parabricks.version="4.7.0-1" \
      io.bio-neb.worker.request-schema="bio-neb.parabricks-worker-request/7" \
      io.bio-neb.worker.receipt-schema="bio-neb.parabricks-worker-receipt/5" \
      io.bio-neb.worker.module="bio_neb.parabricks.reusable_object_storage_parabricks" \
      io.bio-neb.worker.entrypoint="/opt/bio-neb/venv/bin/python /opt/bio-neb/reusable_object_storage_parabricks.py"

# Parabricks 4.7 is Ubuntu 22.04 based.  Keep the derived layer to the worker
# interpreter, its exact compatible object-storage dependencies, and the worker source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bwa=0.7.17-6 \
        python3-venv=3.10.6-1~22.04.1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/bio-neb/venv

COPY containers/parabricks-worker/requirements.txt /opt/bio-neb/requirements.txt
RUN /opt/bio-neb/venv/bin/pip install --no-cache-dir --disable-pip-version-check \
        -r /opt/bio-neb/requirements.txt \
    && rm /opt/bio-neb/requirements.txt

COPY src/bio_neb/parabricks/reusable_object_storage_parabricks.py /opt/bio-neb/reusable_object_storage_parabricks.py
COPY src/bio_neb/parabricks/worker_exchange.py /opt/bio-neb/bio_neb/parabricks/worker_exchange.py
COPY src/bio_neb/contracts.py /opt/bio-neb/bio_neb/contracts.py
COPY src/bio_neb/fasta_selection.py /opt/bio-neb/bio_neb/fasta_selection.py
RUN test -n "${WORKER_SHA256}" \
    && printf '%s  %s\n' "${WORKER_SHA256}" /opt/bio-neb/reusable_object_storage_parabricks.py | sha256sum --check --strict \
    && /opt/bio-neb/venv/bin/python -c "import sys; sys.path.insert(0, '/opt/bio-neb'); import reusable_object_storage_parabricks as worker; worker.worker_source_bytes()"
LABEL io.bio-neb.worker.sha256="${WORKER_SHA256}"

WORKDIR /work
ENTRYPOINT ["/opt/bio-neb/venv/bin/python", "/opt/bio-neb/reusable_object_storage_parabricks.py"]
