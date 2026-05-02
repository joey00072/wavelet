# Wavelet RL WebUI

Lightweight dashboard for the Wavelet orchestrator state server.

## Run

```bash
bun install
bun run dev --host 0.0.0.0
```

Open the Vite URL from your workstation. If the state server is on another
machine, pass the API URL in the page query:

```text
http://<webui-host>:5173/?api=http://<state-server-host>:8765
```

The API base is also editable in the page header and is stored in local storage.

## Backend

The RL state server must be enabled in the RL config:

```yaml
orchestrator:
  state_server:
    enabled: true
    host: 0.0.0.0
    port: 8765
```

The server exposes read-only endpoints and allows browser CORS by default.
