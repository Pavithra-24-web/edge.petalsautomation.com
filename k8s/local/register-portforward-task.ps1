# One-time setup: registers a Windows Scheduled Task that checks every 2
# minutes (starting at login, for the rest of the day) whether the local
# Kubernetes app's port-forwards are up, and (re)starts them if not.
#
# Deliberately does NOT change whether Docker Desktop itself auto-starts -
# the check is a no-op whenever Docker isn't running yet. The point is: you
# open Docker Desktop whenever you want, and within ~2 minutes the app comes
# back up at localhost:3000 on its own, no command to run by hand.
#
# Run this once from a normal (non-admin) PowerShell window:
#   .\k8s\local\register-portforward-task.ps1
#
# To remove it later:
#   Unregister-ScheduledTask -TaskName "PetalEdge Local Port-Forward" -Confirm:$false

$scriptPath = Join-Path $PSScriptRoot "start-portforward.ps1"
$taskName = "PetalEdge Local Port-Forward"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
# Task Scheduler's XML schema rejects [TimeSpan]::MaxValue (duration field
# out of range) - 10 years is effectively "forever" for this purpose and is
# well within the schema's accepted range.
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 2) `
    -RepetitionDuration (New-TimeSpan -Days 3650)).Repetition
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Checks every 2 minutes whether PetalEdge's local Kubernetes port-forwards (backend:8010, frontend:3000) are up, and restarts them if Docker/Kubernetes are ready but the tunnels aren't." `
    -Force

Write-Host "Registered scheduled task '$taskName' - it starts checking at your next login, every 2 minutes."
Write-Host "It's a no-op until you open Docker Desktop; once you do, the app comes back up within ~2 minutes."
Write-Host "To run a check right now: Start-ScheduledTask -TaskName '$taskName'"
