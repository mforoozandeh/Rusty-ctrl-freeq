# Hosting the web application

`.github/workflows/deploy.yml` builds the site on every push to `main` and publishes it to Cloudflare Pages as the
project `rusty-ctrl-freeq`, at <https://rusty-ctrl-freeq.pages.dev>.  It needs two repository secrets (Settings →
Secrets and variables → Actions):

| Secret | What it is |
|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | The account ID shown in the Cloudflare dashboard |
| `CLOUDFLARE_API_TOKEN` | An API token with the *Cloudflare Pages: Edit* permission |

Without them the workflow still builds and keeps the site as the run's `rusty-ctrl-freeq-site` artifact, and skips
the upload.  The first deploy creates the Pages project.

## Hosting elsewhere

`trunk build --release --config gui/Trunk.toml` writes the same static site to `gui/dist`: `index.html`, the
interface's WebAssembly and JavaScript, and the optimisation worker's.  Any static host serves it as it is:

- Asset paths are relative (`public_url = "./"`), so the site works at a domain's root, in a subdirectory, or from
  `python3 -m http.server` in `gui/dist`.
- No special headers are needed.  The optimisation runs in an ordinary Web Worker, not with WebAssembly threads,
  so the cross-origin-isolation headers `SharedArrayBuffer` requires are not.
- Serve `.wasm` as `application/wasm`; `_headers` does that on Cloudflare Pages and Netlify, and most hosts do it by
  default.

If the assets are moved away from `index.html`, set `window.ctrl_freeq_worker_url` in the page to the worker
loader's address.

Point a domain you control at the deployment rather than publishing the host's own address; moving host is then a
DNS change.
