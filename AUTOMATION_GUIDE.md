# Automation Guide: Local Repository to Overleaf Sync

This guide explains how to automate pushing changes from your local repository to GitHub, which will then automatically sync with Overleaf.

## Available Automation Options

### 1. Quick Push Script (`push-to-overleaf.ps1`)
The simplest option for manual but quick synchronization.

**Usage:**
```powershell
# Push with default message
.\push-to-overleaf.ps1

# Push with custom message
.\push-to-overleaf.ps1 -message "Added new chapter on methodology"
```

### 2. Automatic Push After Commit (Git Hook)
The post-commit hook automatically pushes to GitHub after every commit.

**Enable it:**
```powershell
# Make the hook executable (on Git Bash or WSL)
chmod +x .git/hooks/post-commit
```

**Usage:**
```bash
# Just commit normally, push happens automatically
git add .
git commit -m "Your commit message"
# No need to push manually!
```

**To disable:** Delete or rename `.git/hooks/post-commit`

### 3. File Watcher (`watch-and-sync.ps1`)
Monitors your files and automatically commits and pushes changes every 30 seconds.

**Usage:**
```powershell
# Start the watcher
.\watch-and-sync.ps1

# Stop with Ctrl+C
```

**Features:**
- Watches all .tex, .bib, .png, .jpg, and .pdf files
- Batches changes to avoid too frequent pushes
- Automatically creates timestamped commits

### 4. GitHub Actions Workflow
Located in `.github/workflows/workflows.yml`, this automatically builds your PDF when you push to GitHub.

## Recommended Workflows

### For Active Writing Sessions
Use the **File Watcher** (`watch-and-sync.ps1`):
- Start it when you begin writing
- It handles everything automatically
- Stop it when you're done

### For Deliberate Commits
Use the **Git Hook**:
- Make meaningful commits with good messages
- Automatic push ensures Overleaf stays in sync

### For Quick Updates
Use the **Push Script** (`push-to-overleaf.ps1`):
- Perfect for quick fixes
- Control exactly when to sync

## Important Notes

1. **Overleaf Sync Delay**: After pushing to GitHub, Overleaf typically takes 1-2 minutes to sync.

2. **Merge Conflicts**: If you edit on both Overleaf and locally simultaneously, you may encounter conflicts. Always pull before starting local work:
   ```bash
   git pull origin main
   ```

3. **Large Files**: Be mindful of large PDF figures. Consider using Git LFS for files over 50MB.

4. **Security**: Never commit sensitive information. Add a `.gitignore` file if needed.

## Troubleshooting

### PowerShell Execution Policy Error
If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Git Hook Not Working
Ensure the hook file has Unix line endings and is executable:
```bash
dos2unix .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

### File Watcher Not Detecting Changes
- Check that the file extensions are included in the watcher
- Ensure you're saving files (not just editing)
- Try restarting the watcher

## Quick Start

1. For immediate use, try the push script:
   ```powershell
   .\push-to-overleaf.ps1 -message "Testing automation"
   ```

2. For continuous sync during writing:
   ```powershell
   .\watch-and-sync.ps1
   ```

Happy writing! 📝 