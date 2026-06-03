# Security notes

These notes assume the host this project runs on is **DMZ-exposed**
— i.e. the network in front of it does not firewall inbound traffic
to it. (If your host is behind a hardware firewall or router with
strict inbound rules, the public-exposure concerns below are softer,
but the loopback-bind design still matters.) On a DMZ-exposed host
the iptables `INPUT` chain is typically default-DROP plus a few
allowed ports — but **Docker bypasses `INPUT`** by inserting its
own DNAT rules in `nat/PREROUTING` and forwarding in `FORWARD` /
`DOCKER-USER`. So a `-p 8765:8000` publish without an explicit host
bind would make the service world-reachable.

This project takes that seriously.

## What we do

1. **Loopback bind only (by default) — IPv4, literal, runtime-verified.**
   `scripts/03_start_service.sh` publishes the port as
   `-p 127.0.0.1:8765:8000/tcp`. The service is reachable *only* from
   the host's own IPv4 loopback. Any client must be on the same
   machine (or tunnel in over SSH from a separate identity).

   **Opt-out**: running `bash setup.sh --bind-all-interfaces` writes
   `install_state.env` at the project root, overlaying
   `LA_HOST_BIND_IP=0.0.0.0`. Every script in the project then
   publishes on every interface — i.e. the WS / HTTP endpoint is
   reachable from any host that can reach this machine on port 8765.
   This is opt-in only; the default remains loopback. The runtime
   `ss -tlnH` verification below still applies — the kernel-side
   listener must match the chosen bind exactly, otherwise the
   container start aborts. Reverting is `bash setup.sh --bind-loopback`.
   The full set of caveats below ("No TLS", "No auth", "No rate
   limiting") all become live concerns the moment you opt in — read
   them before flipping the flag.

   Three reinforcements:

   - The host bind address (`LA_HOST_BIND_IP`) is the literal
     string `127.0.0.1` in [`scripts/lib/versions.sh`](../scripts/lib/versions.sh).
     We never use `localhost` in any URL or executable code path.
     On many Linux setups (including Ubuntu 24.04 default),
     `getaddrinfo("localhost")` returns `::1` first; a browser or
     curl that does `http://localhost:8765` therefore tries the
     IPv6 loopback, which has no listener, fails with `Connection
     refused`, and produces a confusing "the server is down" error
     when in fact only the name lookup went the wrong way. **All
     client code, all examples, the Docker `docker-compose.yml`, and
     every URL in this repo say `127.0.0.1` literally** — the only
     occurrences of the string `localhost` are in explanatory prose
     warning against it (such as this paragraph).
   - Inside the container the listener does bind to `0.0.0.0:8000`
     (`container/entrypoint.sh`). That is unavoidable: Docker DNATs
     the host's `127.0.0.1:8765` into the container's `eth0`
     interface, not into the container's loopback. A listener on
     `127.0.0.1` *inside* the container would not see forwarded
     traffic and the service would silently appear unreachable.
     The DMZ-safety property comes from the **host-side** publish,
     not the in-container bind.
   - After `docker run`, `scripts/03_start_service.sh` queries the
     host kernel via `ss -tlnH ( sport = :8765 )` and asserts that
     the published port is bound to *exactly* `127.0.0.1:8765` —
     **not** `0.0.0.0:8765`, **not** `[::]:8765`, **not**
     `[::1]:8765`. Any deviation aborts the start with a precise
     diagnostic. The contract is verified at the kernel layer, not
     at the docker-flag layer.

2. **Read-only container root.** `docker run --read-only` plus a
   512 MiB tmpfs on `/tmp` with `noexec,nosuid,nodev` mount flags
   for the Unix socket — anything written into `/tmp` cannot be
   executed and cannot gain privilege via setuid. The model
   directory is bind-mounted **read-only**.

3. **No Linux capabilities.** `--cap-drop=ALL` plus
   `--security-opt=no-new-privileges`. The container cannot gain
   capabilities at runtime.

4. **Non-root inside the container.** A user `la` (UID/GID mapped
   to the desktop user at build time) owns `/opt/locate_anything`.
   `root` is unreachable from inside.

5. **Validated input at the network boundary.** The Rust frontend
   validates every JPEG before passing it to the Python worker:
   magic bytes, declared vs. actual payload size, header-only
   dimension parse against `LA_MAX_IMAGE_DIM`. A malformed image is
   rejected at the edge — a framing-level violation closes the
   WebSocket (`Close 1008`), a per-frame image fault returns
   `type:"error"` with `code:"invalid_image"` — and does NOT reach
   the Python decoder.

6. **No external network access from the worker.** The Python
   sidecar does not call out to the internet at runtime. All
   weights are downloaded once during setup via the host script
   (`scripts/01_download_weights.sh`) and bind-mounted into the
   container.

7. **No fallback paths.** If `flash-attn` fails to import, the
   container exits non-zero — it does NOT silently fall through to
   SDPA. If the GPU compute capability is wrong, ditto. This is a
   safety property as well as a correctness property: an unexpected
   degraded mode at runtime is the kind of thing operators don't
   notice.

## What we do NOT do

* **TLS termination.** The server speaks plain HTTP / `ws://` on
  loopback. This is deliberate: the loopback path never crosses
  any network adapter (not a NIC, not a bridge — `127.0.0.1` is
  handled entirely inside the kernel's `lo` driver), so an
  on-the-wire encryption layer would add CPU cost and key
  management with no eavesdropping surface to protect against.
  `axum::serve(listener, app)` is plain HTTP — there is no
  rustls bind, no `wss://` handler. Use SSH local-forwarding or
  a reverse proxy with TLS in front if you ever need
  cross-host access.

* **Authentication.** Anyone on the host's loopback can hit
  `/v1/stream`. The implied threat model is: only the
  legitimate user has shell on the host. If you want auth, run
  it behind nginx with a `client_certificate` block or stick a
  small auth proxy in front.

* **Rate limiting.** Each WebSocket is strictly stop-and-wait, so
  per-connection rate is bounded by the GPU's service time — TCP
  flow control applies backpressure naturally. There is no global
  connection limiter; multiple connections from the same client
  serialize on the GPU's asyncio.Lock in the Python sidecar, which
  is the only fairness mechanism. If you publish this to multi-tenant
  traffic, add a proper rate limiter upstream.

* **Audit logging of payloads.** The structured logs (JSON to
  stdout) log frame counts and latencies, not image contents or
  prompts. If you want full request auditing, change the
  `tracing` configuration in `rust_server/src/main.rs`.

## Threats this design does NOT defend against

* **Trojan model weights at the upstream.** We pin to a specific full
  40-char HF commit (`LA_MODEL_HF_REVISION`) and SHA-256-verify every
  shipped `.py`, every safetensors shard, and every inference-relevant
  config/tokenizer file at boot AND immediately before
  `from_pretrained` runs (`worker/inference.py` re-calls
  `validate_model_dir` to shrink the TOCTOU window to sub-millisecond).
  We also reject any unpinned `.py` file in the model directory (defends
  against new `__init__.py` injection that `trust_remote_code` would
  transitively import). But: we do not verify the upstream weights
  against a signed manifest. If `nvidia/LocateAnything-3B` is silently
  re-pushed under the same commit SHA (HF's content-addressed storage
  doesn't allow that in practice, but…), we'd not catch it.

* **A host-user attacker who can write to `./models/LocateAnything-3B/`
  during operation.** The bind-mount is read-only INSIDE the container,
  but the underlying host filesystem is writable by the user who ran
  `setup.sh`. That user is already root-equivalent on the box via the
  docker group, so this isn't a meaningful escalation, but for the
  paranoid: `chown root:root -R ./models/LocateAnything-3B/ && chmod
  -R a-w ./models/LocateAnything-3B/` after running
  `01_download_weights.sh` removes write access from the desktop user
  entirely. The container's `la` user can still read the files.

* **Untrusted JPEG decoder bugs in libjpeg-turbo / PIL.** We rely on
  the OS's libjpeg-turbo and PIL for image decoding. CVE-grade bugs
  in those libraries would be exploitable through this server. The
  Rust frontend's header-only parse via `jpeg-decoder` adds a
  layer of validation but does not replace the full decode.

* **Side-channel attacks against GPU compute.** Out of scope.

## Operational recommendations

* Run the service as the desktop user; the docker group is
  effectively root-equivalent (any user in `docker` can mount
  arbitrary host paths into a container), so on a security-
  sensitive host either keep that group tight or run the
  container under a dedicated unprivileged account.

* If you ever want to expose this service to the LAN: run
  `bash setup.sh --bind-all-interfaces` (the persistent opt-in
  flag — see §"What we do" / 1 above) **and** put it behind a
  reverse proxy with TLS + an auth scheme before letting any
  untrusted client reach port 8765. Without the proxy, anyone
  reachable on that port can drive your GPU.

* Watch `docker stats locate-anything` during heavy use — if memory
  usage approaches the host's VRAM limit, the Python worker will
  start emitting CUDA OOM errors. The server reports those as a
  per-frame `type:"error"` with `code:"internal"` (the message starts
  `"CUDA out of memory"`), followed by a `Close(1011)` as the worker
  self-exits to restore CUDA state; clients should back off and
  reconnect after the ~10–15 s restart window.
