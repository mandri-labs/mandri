FROM node:24-slim

ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-httpx ca-certificates git openssh-client ripgrep iptables util-linux procps bubblewrap \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code@2.1.296 @openai/codex@0.162.1 @opencode/cli@2.0.26 @earendil-works/pi-coding-agent@1.1.0 \
    && npm cache clean --force
ADD https://storage.googleapis.com/antigravity-public/antigravity-cli/1.3.3-5524738307653632/linux-x64/cli_linux_x64.tar.gz /tmp/agy.tar.gz
RUN test "$TARGETARCH" = amd64 \
    && echo "abe3ddf5cf30d3adaa586acf505221aa19702f5ca71af3c8b1f9b1a5976bd8e5c151c41ea33b39c41675d25843462a3e8c382c054f580e9d7e1688919d3b5110  /tmp/agy.tar.gz" | sha512sum --check --strict \
    && tar -xzf /tmp/agy.tar.gz -C /usr/local/bin antigravity \
    && mv /usr/local/bin/antigravity /usr/local/bin/agy \
    && chmod 755 /usr/local/bin/agy \
    && rm /tmp/agy.tar.gz
COPY docker/worker/worker.py /opt/mandri/worker.py
COPY mandri_runtime/src/mandri/runtime/agy_hook.py /opt/mandri/agy_hook.py
COPY mandri_core/src/mandri/core/resources/pi_gateway.ts /opt/mandri/pi_gateway.ts
COPY mandri_core/src/mandri/core/resources/pi_managed.ts /opt/mandri/pi_managed.ts
COPY mandri_core/src/mandri/core/resources/pi_permissions.ts /opt/mandri/pi_permissions.ts
RUN mkdir -p /workspace /home/worker \
    && chown -R 1000:1000 /workspace /home/worker \
    && chmod 755 /opt/mandri/*.py
ENV HOME=/home/worker PATH=/usr/local/bin:/usr/bin:/bin CI=1 DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 OPENCODE_AUTOUPDATE=false AGY_CLI_DISABLE_AUTO_UPDATE=true
LABEL io.mandri.worker.version="1" io.mandri.worker.harnesses="codex,claude,opencode,agy,pi"
LABEL io.mandri.worker.workspace-identity="1"
LABEL io.mandri.worker.native-run="1"
WORKDIR /workspace
USER 1000:1000
ENTRYPOINT ["python3", "/opt/mandri/worker.py", "run"]
