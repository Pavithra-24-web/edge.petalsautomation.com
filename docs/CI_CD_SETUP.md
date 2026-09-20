# CI/CD → Linode Kubernetes (LKE) setup

The intended day-to-day flow: work happens on `dev`, gets reviewed as a pull
request, and only reaches `main` — and therefore production — once you
approve it on GitHub:

1. Push to `dev` while you're working. This runs backend tests (pytest) and
   a frontend lint+build automatically, so you get fast feedback — it does
   **not** build/push images or touch Linode.
2. If tests pass and `dev` has commits `main` doesn't, the `open-pr` job
   **automatically opens a pull request from `dev` into `main`** (or leaves
   the existing one alone if one's already open — new commits just show up
   on it). Nothing to run by hand.
3. **You review and approve the PR on GitHub, then merge it.** This is a
   manual gate by design — nothing deploys without your sign-off. (Requires
   a one-time branch protection setting — see **Requiring PR review**
   below; a workflow file can't enforce this by itself.)
4. Merging into `main` builds Docker images for the backend (API + Celery
   worker share one image) and frontend, pushes them to **GitHub Container
   Registry (GHCR)** — `ghcr.io/<owner>/<repo>-backend` and `-frontend`,
   tagged with the commit SHA and `latest` — and deploys them to a **Linode
   Kubernetes Engine (LKE)** cluster using the plain YAML manifests in
   [`k8s/`](../k8s).

## Known issue: test-backend doesn't block the pipeline yet

`backend/tests/` had never actually been run in a clean environment before
this pipeline existed (it wasn't even committed to git until now). Doing so
for the first time surfaced **~73 failing/erroring tests out of ~267** —
stale assertions from before the app was renamed (e.g. `test_root` expects
`"EdgeImpulse"`, the code now returns `"petaledge API"`), missing mocks
(some tests make real `smtplib` calls), and a cluster of FOMO/SSD/YOLO
deployment-routing failures that needs real investigation, not a quick
guess. That's a separate, substantial cleanup effort from standing up this
pipeline, so `test-backend` currently runs with `continue-on-error: true` —
it still runs and reports its real pass/fail status, but a failure doesn't
block PRs from opening or merges from deploying. Remove that once the suite
is actually green.

Workflow file: [`.github/workflows/ci-cd.yml`](../.github/workflows/ci-cd.yml).

## Requiring PR review (one-time GitHub setting)

This has to be set on github.com — it's a repository policy, not something a
workflow YAML file can express:

1. Go to the repo on GitHub → **Settings → Branches**.
2. Under **Branch protection rules**, click **Add rule** (or **Add branch
   ruleset**, depending on your GitHub UI version).
3. Branch name pattern: `main`.
4. Check **Require a pull request before merging**.
5. Check **Require approvals** (1 is enough for a small team).
6. Save. Direct pushes to `main` are now rejected for everyone — including
   you — until a PR gets an approval and is merged.

This is a first pass, sized for one environment (production) with everything
— Postgres, Redis, MinIO — running inside the cluster, matching
`docker-compose.yml`. The GPU Celery worker (`make worker-gpu`) is
intentionally not included; there's no GPU node pool yet.

---

## 1. Create the LKE cluster

In [Linode Cloud Manager](https://cloud.linode.com/kubernetes/clusters) →
**Create Cluster**:

- Pick a region close to your users.
- Node pool: start with **2× Linode 4GB** nodes. This app's images include
  TensorFlow/PyTorch/OpenCV, so give it real memory — 2GB nodes will struggle.
- Leave HA control plane off for now (it costs extra and isn't needed for a
  first setup).

Create it, then wait a few minutes for the nodes to go `Ready`.

## 2. Get kubectl access from GitHub Actions

From the cluster's page in Cloud Manager, download the **kubeconfig**.

The workflow expects it base64-encoded in a secret called `KUBE_CONFIG`.
From a machine with the file:

```bash
# macOS
base64 -i kubeconfig.yaml | pbcopy
# Linux
base64 -w0 kubeconfig.yaml | xclip -selection clipboard
# Windows PowerShell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("kubeconfig.yaml")) | Set-Clipboard
```

Then in GitHub: **Settings → Secrets and variables → Actions → Secrets →
New repository secret** → name it `KUBE_CONFIG`, paste the base64 value.

You do **not** need a separate registry account or token: the workflow pushes
to GHCR and pulls into the cluster using GitHub's own built-in
`GITHUB_TOKEN`, scoped to this repo's packages.

## 3. Add the application secrets

The workflow creates the in-cluster `app-secrets` Secret from these GitHub
Actions **secrets** (Settings → Secrets and variables → Actions → Secrets):

| GitHub secret name        | Used for                                          |
|----------------------------|---------------------------------------------------|
| `APP_SECRET_KEY`           | app session/signing key — `python -c "import secrets; print(secrets.token_hex(32))"` |
| `APP_JWT_SECRET`           | JWT signing key — generate the same way, a **different** value |
| `APP_POSTGRES_PASSWORD`    | Postgres password (pick any strong value; first deploy creates the DB with it) |
| `APP_S3_ACCESS_KEY`        | MinIO access key (pick any strong value)          |
| `APP_S3_SECRET_KEY`        | MinIO secret key (pick any strong value)          |
| `APP_GOOGLE_CLIENT_SECRET` | only if you use Google OAuth server-side; blank is fine otherwise |
| `APP_SMTP_USERNAME`        | Gmail SMTP username, for verification emails; blank disables email |
| `APP_SMTP_PASSWORD`        | Gmail App Password                                |

Everything non-secret (hostnames, ports, feature flags) lives in
[`k8s/configmap.yaml`](../k8s/configmap.yaml) and is committed to the repo.

## 4. First deploy

Push to `main` (or merge this branch into it). Watch the run in the
**Actions** tab. It will:

- run tests,
- build + push both images,
- apply everything under `k8s/`,
- run `alembic upgrade head` as a one-off Job,
- roll out `backend`, `worker-cpu`, and `frontend`.

## 5. Bootstrap step: wire up the external addresses

`backend`, `frontend`, and `minio-external` are all `type: LoadBalancer`
Services — Linode's cloud-controller-manager gives each one its own external
IP automatically (each is a small monthly cost — see **Costs** below). Two
things in this repo reference those addresses *before they exist*, so one
follow-up round is needed after the very first deploy:

```bash
kubectl get svc -n petaledge
# note the EXTERNAL-IP for backend, frontend, and minio-external
```

1. **Frontend's API URL** — `NEXT_PUBLIC_API_URL` / `NEXT_PUBLIC_WS_URL` are
   baked into the frontend's static JS at *build time* (it's a static
   export, calling the API straight from the browser). Set them as GitHub
   Actions **variables** (not secrets — they end up in public client code):
   Settings → Secrets and variables → Actions → **Variables** tab:
   - `NEXT_PUBLIC_API_URL` = `http://<backend EXTERNAL-IP>`
   - `NEXT_PUBLIC_WS_URL` = `ws://<backend EXTERNAL-IP>`
   - `NEXT_PUBLIC_GOOGLE_CLIENT_ID` = your OAuth client ID, if used

2. **Backend's public config** — edit
   [`k8s/configmap.yaml`](../k8s/configmap.yaml) and replace the two
   `CHANGE-ME` placeholders:
   - `S3_PUBLIC_ENDPOINT` → `http://<minio-external EXTERNAL-IP>:9000`
   - `FRONTEND_URL` / `CORS_ORIGINS` → `http://<frontend EXTERNAL-IP>`

Commit the configmap change and push to `main` again — this re-triggers the
pipeline, rebuilding the frontend with the real API URL and reapplying the
updated config. (Once you point a real domain at these IPs, use the domain
instead and get TLS via an Ingress — see **Next steps**.)

## Costs to be aware of

- LKE control plane: free (non-HA).
- Each node in the pool: billed hourly like a regular Linode.
- Each `LoadBalancer` Service: provisions a Linode NodeBalancer (~$10/month).
  This setup has three (`backend`, `frontend`, `minio-external`) — see
  **Next steps** for how to collapse that to one later.

## Day-2 operations

```bash
kubectl get pods -n petaledge                 # is everything Running?
kubectl logs -n petaledge deploy/backend -f   # tail backend logs
kubectl logs -n petaledge deploy/worker-cpu -f
kubectl get svc -n petaledge                  # external IPs
kubectl rollout undo deployment/backend -n petaledge   # roll back a bad deploy
```

Every push to `main` redeploys automatically. There's no manual `kubectl
apply` needed for routine changes — just merge to `main`.

## Next steps (not done in this first pass)

- **Ingress + TLS**: install `ingress-nginx` and `cert-manager` to get one
  shared LoadBalancer with HTTPS instead of three plain-HTTP ones.
- **GPU worker**: add a GPU node pool and a `worker-gpu` Deployment
  (`nodeSelector`/`tolerations`) once training workloads need it.
- **Managed Postgres/object storage**: Linode offers managed databases and
  S3-compatible Object Storage — swapping to those removes the
  `postgres`/`minio` PVCs as single points of failure.
- **Separate staging environment**: duplicate the `petaledge` namespace (e.g.
  `petaledge-staging`) and branch the workflow on `develop` vs `main` once
  you want a pre-production environment.
