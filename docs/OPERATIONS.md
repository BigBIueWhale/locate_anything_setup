# Operations — lifecycle, restart, offline, uninstall

This document covers the day-to-day operational lifecycle of the
service: what runs where, how to restart, what survives a reboot,
what survives the internet falling over, and how to clean up.

For first-time setup, see [`../README.md`](../README.md).

---

## Lifecycle overview

```
┌── setup.sh ──────────────────────────────────────────┐
│  00 validate host                                    │
│  01 download weights + generate calibration image    │
│  02 docker build (skipped if image already tagged)   │
│  03 docker run (restart unless-stopped)              │
│  04 smoke test                                       │
└──────────────────────────────────────────────────────┘
              │
              ▼ (container running, --restart unless-stopped)
┌── steady state ──────────────────────────────────────┐
│  docker daemon supervises the container.             │
│  Crash → restart. Reboot → restart on next boot.     │
│  No network calls at runtime — fully self-contained. │
└──────────────────────────────────────────────────────┘
              │
              ▼ (operator action)
┌── stop / start (no setup re-run) ────────────────────┐
│  docker stop  locate-anything     # graceful stop    │
│  docker start locate-anything     # back up          │
└──────────────────────────────────────────────────────┘
              │
              ▼ (operator action)
┌── uninstall ─────────────────────────────────────────┐
│  bash uninstall.sh [--remove-image] [--remove-weights│
│                    [--remove-hf-cache]               │
│                    [--remove-rust-target] [--purge]  │
└──────────────────────────────────────────────────────┘
```

---

## Restart policy semantics — read this once

The container runs with `--restart unless-stopped`. **Important
nuance**: Docker's restart policies trigger on **container EXIT**, not
on `unhealthy` health status. Concretely:

| Event                                                          | Restart fires? |
|----------------------------------------------------------------|----------------|
| Python worker crash (validation fail, OOM, segfault)           | **YES** — entrypoint.sh's `wait -n` returns, peer is signalled, container exits. |
| Rust binary crash (panic with `panic = "abort"`)               | **YES** — same path. |
| Host reboot                                                    | **YES** — Docker starts the container on boot. |
| Healthcheck returns `unhealthy` (worker wedged, lock stuck)    | **NO**         |

If the worker is alive but wedged (e.g., a CUDA hang doesn't kill the
process; the asyncio loop is stuck behind the model lock), the
container will be reported as `unhealthy` but **will not be restarted
automatically** by Docker. Recovery requires `docker restart
locate-anything` or attaching an external "autoheal" sidecar.

Why we don't bake autoheal into the project:

- The healthcheck is *deep* (`/v1/health` does a real round-trip
  through the worker's `info` path) — true wedges should manifest as
  the asyncio loop being unable to respond within 5s, and the
  process exits via the existing
  `worker/validate_startup.py`/`la_worker.py` paths in most failure
  modes.
- Autoheal sidecars (the standard one is
  [`willfarrell/autoheal`](https://github.com/willfarrell/docker-autoheal))
  add an extra always-running root-equivalent container, which we'd
  rather not add to a single-user setup.
- The Docker socket access required for autoheal is itself a security
  consideration.

If you observe wedges in practice, two options:

1. Add `willfarrell/autoheal` as a sidecar in `docker-compose.yml`
   with `--label autoheal=true` on this container.
2. Bake a tiny in-container watchdog into the entrypoint that polls
   `/v1/health` from inside and SIGTERMs PID 1 after N consecutive
   failures.

Neither is enabled by default. Operators should monitor `docker ps
--filter health=unhealthy` if running unattended.

## What survives what

| Event                                | Container | Image | Weights | HF cache |
|---|---|---|---|---|
| `docker stop locate-anything`        | paused    | yes   | yes     | yes      |
| Host reboot                          | restarted automatically | yes | yes | yes |
| `docker rm -f locate-anything`       | gone      | yes   | yes     | yes      |
| `bash uninstall.sh`                  | gone      | yes   | yes     | yes      |
| `bash uninstall.sh --remove-image`   | gone      | gone  | yes     | yes      |
| `bash uninstall.sh --remove-weights` | gone      | yes   | gone    | yes      |
| `bash uninstall.sh --purge`          | gone      | gone  | gone    | gone     |

The service uses `docker run --restart unless-stopped`, so a reboot
brings it back without any manual step. Use `docker stop
locate-anything` explicitly if you want the service to stay down
across a reboot.

---

## Offline operation

After a successful `setup.sh` run, **the service runs entirely
offline**:

- The Docker image is built locally and stored in
  `/var/lib/docker/`. No further pulls happen at runtime.
- Model weights are bind-mounted from `./models/LocateAnything-3B/`.
- The Python worker sets `HF_HUB_DISABLE_TELEMETRY=1` and never
  reaches HuggingFace at runtime.
- The Rust frontend opens only the TCP listener on
  `${LA_HOST_BIND_IP}:${LA_HOST_PORT}` and the Unix socket to the
  Python worker. No outbound connections.

### Re-running `setup.sh` while offline

`setup.sh` is idempotent and offline-safe **once everything has been
installed once**:

| Step | Offline-safe re-run? | Why |
|---|---|---|
| 00 validate host | yes if the nvidia/cuda base image is locally cached (uses `--pull=never` after first run) | `docker image inspect` test |
| 01 download weights | yes if `./models/LocateAnything-3B` already has the safetensors | size check short-circuits |
| 02 docker build | yes if `LA_IMAGE_TAG` is already locally built | `docker image inspect` short-circuit (force with `--rebuild`) |
| 03 start service | yes — only does `docker run` from the local image | nothing networked |
| 04 smoke test | yes — runs `docker exec` inside the already-built image; no `pip install`, no helper container | |

**Verification you should do after a reboot with no network**:

```bash
docker ps           # locate-anything should be Up
curl -s http://127.0.0.1:8765/v1/health         | jq .   # status: ok
curl -s http://127.0.0.1:8765/v1/capabilities   | jq .   # calibration block present
bash setup.sh                                            # should print "skipping" for each step
```

If the third command tries to pull something over the network, you've
found a bug — open an issue.

---

## Uninstall — the contract

[`uninstall.sh`](../uninstall.sh) is the only destructive script in
the project. Its contract:

1. **Default = instances-only.** With no flags, it stops the
   container and removes the container instance. Nothing else is
   touched — the Docker image, the model weights, and the HF cache
   all survive. This lets you "uninstall the service" while
   preserving the slow-to-rebuild parts (the ~12 GiB image and the
   ~7.66 GiB weights).

2. **Granular flags.** Each destructive operation has its own opt-in
   flag:

   - `--remove-image`       — `docker image rm ${LA_IMAGE_TAG}`
   - `--remove-weights`     — `rm -rf ./models/LocateAnything-3B` and `./test_data/calibration.jpg`
   - `--remove-hf-cache`    — `rm -rf ./cache/huggingface`
   - `--remove-rust-target` — `rm -rf ./rust_server/target`
   - `--purge`              — all of the above
   - `--yes` / `-y`         — skip the confirmation prompt

3. **No silent failures.** Each step runs three checks:
   1. *Inspect* whether the target exists (`docker ps -a`,
      `docker image inspect`, `test -d`, etc.).
   2. If present, run the removal command (`docker stop`,
      `docker rm`, `docker image rm`, `rm -rf`).
   3. *Assert* the post-condition (the target is now gone). On
      failure, the script aborts with a precise diagnostic. There
      is no `|| true` or "best-effort" path.

4. **Docker daemon down → loud refusal, not skip.** If
   `docker info` fails, the script warns (since it can't operate on
   any container or image) and exits — `--remove-weights` and
   `--remove-rust-target` still work since they're filesystem-only.

5. **BuildKit cache is preserved.** The shared BuildKit layer cache
   under `/var/lib/docker/buildkit/` is NOT pruned, since it
   accelerates rebuilds of unrelated projects on the same host.
   Free it manually with `docker builder prune` when needed.

### What `uninstall.sh` does NOT touch

- **Host packages** (driver, CUDA toolkit, Docker, nvidia container
  toolkit). Those were installed by the personal_server scripts and
  are this project's *prerequisites*, not its dependencies. Uninstall
  them via apt if you need to.

- **The project source tree**. Source files, `README.md`, scripts,
  `versions.sh`, `Cargo.toml`, `Cargo.lock`. The source itself is
  considered "what you cloned" — removing it is a `git`/`rm`
  decision for the operator.

- **`scripts/lib/versions.sh`**. The pin file is the single source of
  truth for what would be installed if you re-ran `setup.sh`; we never
  remove it.

---

## Useful one-liners

```bash
# tail the service logs
docker logs -f --tail 200 locate-anything

# inspect the published port mapping
docker port locate-anything

# trigger a model self-test from the host
docker exec locate-anything python /opt/locate_anything/scripts/lib/smoke_ws_client.py \
    --url "ws://127.0.0.1:8000/v1/stream" \
    --image /opt/locate_anything/test_data/calibration.jpg \
    --prompt 'Locate all the instances that matches the following description: person.' \
    --mode hybrid --timeout 60

# image and weight sizes on disk
docker image inspect locate-anything:la3b-cu130-torch2.12-fa2.8.4 --format '{{.Size}}' | numfmt --to=iec
du -sh ./models ./cache ./rust_server/target 2>/dev/null
```
