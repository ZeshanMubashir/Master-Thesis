# PowerShell script to quickly push changes to GitHub (and sync to Overleaf)
param(
    [string]$message = "Update thesis content"
)

Write-Host "Pushing changes to GitHub (will sync to Overleaf)..." -ForegroundColor Green

# Stage all changes
git add .

# Show status
git status --short

# Commit with provided message
git commit -m $message

# Push to GitHub
git push origin main

Write-Host "Changes pushed successfully! Overleaf will sync shortly." -ForegroundColor Green 