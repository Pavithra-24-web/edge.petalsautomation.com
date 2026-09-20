# Running PetalEdge on Kubernetes locally

This is purely for trying the app out on Kubernetes on your own machine — no
GitHub Actions, no Linode, no registry. You build the images yourself and
apply the manifests in [`k8s/local/`](../k8s/local) by hand with `kubectl`.

## 1. Install Docker Desktop with Kubernetes

You currently have neither Docker nor kubectl installed. The simplest path
on Windows is **Docker Desktop**, which bundles both a Kubernetes cluster and
the `kubectl` CLI in one install:

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   (requires WSL2 — the installer will prompt you to enable it if needed).
2. Open Docker Desktop → **Settings → Kubernetes** → check **Enable
   Kubernetes** → **Apply & Restart**. Wait for the green "Kubernetes
   running" indicator (first start can take a few minutes, it's pulling
   cluster images).
3. Verify from a terminal:
   ```
   docker --version
   kubectl version --client
   kubectl config current-context   # should print "docker-desktop"
   ```

(Alternative: `minikube` works too, but needs an extra `minikube image load`
step after every build since it runs in its own VM instead of sharing
Docker Desktop's image store. Docker Desktop is simpler to start with.)

## 2. Build the images locally

From the repo root:

```bash
# Backend's build context is the repo root, not backend/ — its Dockerfile
# also needs the sibling unoq/ directory (embedded into .pxe device packages).
docker build -t petaledge-backend:local -f backend/Dockerfile .

docker build -t petaledge-frontend:local \
  --build-arg NEXT_PUBLIC_API_URL=http://localhost:8010 \
  --build-arg NEXT_PUBLIC_WS_URL=ws://localhost:8010 \
  frontend
```

The frontend build-args matter: it's a static export, so the API URL gets
baked into the JS at build time — set it to where you'll reach the backend
in step 4 (`localhost:8010`, via `kubectl port-forward`).

Because Docker Desktop's Kubernetes shares the same Docker engine you just
built with, these images are immediately visible to the cluster — no push,
no registry.

## 3. Apply the manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/local/secret.yaml
kubectl apply -f k8s/local/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/minio.yaml
kubectl apply -f k8s/local/migrate-job.yaml
kubectl apply -f k8s/local/backend-deployment.yaml
kubectl apply -f k8s/local/worker-cpu-deployment.yaml
kubectl apply -f k8s/local/frontend-deployment.yaml
```

Watch it come up:

```bash
kubectl get pods -n petaledge -w
```

Everything should reach `Running` (`postgres`/`redis`/`minio` first, then
`migrate-local` runs to `Completed`, then `backend`/`worker-cpu`/`frontend`
go `Running`). Ctrl+C once they're all healthy.

If something sits in `Pending` or `CrashLoopBackOff`:

```bash
kubectl describe pod -n petaledge <pod-name>   # why it won't schedule/start
kubectl logs -n petaledge <pod-name>           # what it printed before dying
```

## 4. Reach it from your browser

These Services are `ClusterIP` (internal-only), so forward their ports to
your machine:

```bash
kubectl port-forward -n petaledge svc/backend 8010:80
kubectl port-forward -n petaledge svc/frontend 3000:80
kubectl port-forward -n petaledge svc/minio 9000:9000   # only if testing uploads
```

Each of those blocks a terminal — run them in separate tabs, or background
them. Then open **http://localhost:3000**.

**Note:** `kubectl port-forward` is just a local tunnel process, not a
persistent service — it dies whenever Docker Desktop restarts (including a
normal machine reboot), even though the pods themselves keep running. Rather
than re-running the commands above by hand every time, set up
[`k8s/local/start-portforward.ps1`](../k8s/local/start-portforward.ps1) once:

```powershell
.\k8s\local\register-portforward-task.ps1
```

This registers a Windows Scheduled Task that checks every 2 minutes whether
the tunnels are up and (re)starts them if Docker/Kubernetes are ready but
they aren't — it doesn't change whether Docker Desktop itself auto-starts,
it just means the app comes back at localhost:3000 within ~2 minutes of you
opening Docker Desktop, without running anything by hand. Remove it later
with `Unregister-ScheduledTask -TaskName "PetalEdge Local Port-Forward"`.

## 5. Making changes and re-testing

Every code change needs a rebuild + restart (there's no hot-reload in a
container image):

```bash
docker build -t petaledge-backend:local -f backend/Dockerfile .
kubectl rollout restart deployment/backend -n petaledge
```

(For everyday backend/frontend development, `make backend` / `make frontend`
— running the app directly on your machine, per the Makefile — is still much
faster than this loop. Use this local-Kubernetes setup specifically to
sanity-check the Kubernetes manifests themselves, not for day-to-day coding.)

## 6. Tear down

```bash
kubectl delete namespace petaledge
```

This deletes everything in one shot, including the PersistentVolumeClaims
(so Postgres/Redis/MinIO data is gone too — expected for a throwaway local
test).

---

Once this works and you're comfortable with what Kubernetes is doing, the
real deployment target is Linode via the GitHub Actions pipeline — see
[`docs/CI_CD_SETUP.md`](CI_CD_SETUP.md). Nothing here talks to GitHub; it's
entirely local and separate from that pipeline.
