# Idempotent health-check-and-heal for the local Kubernetes port-forwards
# (localhost:8010 -> api, localhost:3000 -> web).
#
# kubectl port-forward is just a local tunnel process, not a persistent
# service - it dies on every Docker Desktop restart even though the pods
# themselves keep running. This script does nothing if both tunnels already
# respond; otherwise it waits (briefly - it's designed to be called
# repeatedly, not to block for a long time) for Docker/Kubernetes/the pods,
# then (re)starts the tunnels.
#
# Meant to be run on a short repeating schedule (see
# register-portforward-task.ps1), not just once at login - that way, as soon
# as you manually open Docker Desktop, the next check picks it up and brings
# the app back up on its own, with no autostart change to Docker Desktop
# itself and no need to re-run anything by hand.

$logFile = "$env:TEMP\petaledge-portforward.log"
function Log($msg) {
    "$(Get-Date -Format o)  $msg" | Out-File -Append -FilePath $logFile
}

function Test-Url($url) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

# Already healthy? Don't touch working tunnels.
if ((Test-Url "http://localhost:8010/health") -and (Test-Url "http://localhost:3000")) {
    exit 0
}

Log "=== tunnels down, checking whether to (re)start ==="

# Docker not even running yet (e.g. you haven't opened Docker Desktop this
# session) - nothing to do, the next scheduled check will try again.
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Log "Docker isn't running yet - will check again next cycle."
    exit 0
}

# Docker's up but Kubernetes might still be initializing right after it starts.
kubectl get nodes *> $null
if ($LASTEXITCODE -ne 0) {
    Log "Docker is up but the Kubernetes API isn't ready yet - will check again next cycle."
    exit 0
}

Log "Docker + Kubernetes are ready - proceeding."

# Clear out any stale/half-dead port-forward processes before starting fresh ones.
Get-CimInstance Win32_Process -Filter "Name = 'kubectl.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "port-forward" -and $_.CommandLine -match "petaledge" } |
    ForEach-Object {
        Log "Stopping stale port-forward pid $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

kubectl wait --for=condition=ready pod -l app=api -n petaledge --timeout=90s *>> $logFile
kubectl wait --for=condition=ready pod -l app=web -n petaledge --timeout=90s *>> $logFile

Start-Process -WindowStyle Hidden -FilePath "kubectl" -ArgumentList "port-forward -n petaledge svc/api 8010:80"
Start-Process -WindowStyle Hidden -FilePath "kubectl" -ArgumentList "port-forward -n petaledge svc/web 3000:80"

Log "Port-forwards started: api -> localhost:8010, web -> localhost:3000"
