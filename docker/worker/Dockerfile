FROM node:24-slim

ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-httpx ca-certificates git openssh-client ripgrep iptables util-linux procps bubblewrap \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code@2.1.282 @openai/codex@0.159.0 @opencode/cli@2.0.22 @earendil-works/pi-coding-agent@0.87.1 \
    && npm cache clean --force
ADD https://storage.googleapis.com/antigravity-public/antigravity-cli/1.2.11-6016716732497920/linux-x64/cli_linux_x64.tar.gz /tmp/agy.tar.gz
RUN test "$TARGETARCH" = amd64 \
    && echo "ca12c262343f29a2b87423d1ff1e4244989e936e37fe1f8e56056b0f91cd02f93f133321229ce35c86c2d84b977e919937a3fd9430cfd769a16d6b03ede25081  /tmp/agy.tar.gz" | sha512sum --check --strict \
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
WORKDIR /workspace
USER 1000:1000
ENTRYPOINT ["python3", "/opt/mandri/worker.py", "run"]
