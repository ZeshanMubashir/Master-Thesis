# PowerShell File Watcher for automatic Overleaf sync
# This script monitors LaTeX files and automatically pushes changes to GitHub

Write-Host "Starting file watcher for automatic Overleaf sync..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor Yellow

# Define files to watch
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $PSScriptRoot
$watcher.Filter = "*.*"
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# Define which files to monitor
$extensions = @('.tex', '.bib', '.png', '.jpg', '.pdf')

# Track last push time to avoid too frequent pushes
$script:lastPush = Get-Date
$script:pendingChanges = $false
$pushInterval = 30 # seconds

# Function to push changes
function Push-Changes {
    if ($script:pendingChanges) {
        Write-Host "`nDetected changes, pushing to GitHub..." -ForegroundColor Cyan
        
        # Stage all changes
        git add .
        
        # Commit with timestamp
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit -m "Auto-sync: $timestamp"
        
        # Push to GitHub
        git push origin main
        
        Write-Host "Changes pushed to GitHub! Overleaf will sync." -ForegroundColor Green
        
        $script:lastPush = Get-Date
        $script:pendingChanges = $false
    }
}

# Define action when file changes
$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    $extension = [System.IO.Path]::GetExtension($path)
    
    # Only process relevant files
    if ($extension -in $extensions) {
        $timeStamp = $Event.TimeGenerated
        Write-Host "File changed: $path at $timeStamp" -ForegroundColor Gray
        $script:pendingChanges = $true
    }
}

# Register event handlers
Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Deleted" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Renamed" -Action $action

# Main loop - check for pending changes every few seconds
try {
    while ($true) {
        Start-Sleep -Seconds 5
        
        # Check if enough time has passed since last push
        $timeSinceLastPush = (Get-Date) - $script:lastPush
        if ($timeSinceLastPush.TotalSeconds -ge $pushInterval) {
            Push-Changes
        }
    }
} finally {
    # Clean up
    Get-EventSubscriber | Unregister-Event
    $watcher.Dispose()
    Write-Host "`nFile watcher stopped." -ForegroundColor Red
} 