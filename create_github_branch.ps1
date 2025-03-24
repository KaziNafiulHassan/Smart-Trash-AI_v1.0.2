# Create GitHub Branch Script for Neo4j Migration (PowerShell)

Write-Host "=== Smart Trash AI - GitHub Branch Creator ===" -ForegroundColor Cyan
Write-Host "This script will help you create and push a branch for the Neo4j migration."

# Check if git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git is not installed. Please install git first." -ForegroundColor Red
    exit 1
}

# Check if current directory is a git repository
if (!(Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    git add .
    git commit -m "Initial commit"
}

# Create and switch to a new branch
$BRANCH_NAME = "neo4j-migration"
Write-Host "Creating and switching to branch: $BRANCH_NAME" -ForegroundColor Yellow
git checkout -b $BRANCH_NAME

# Add all files
git add .

# Commit changes
git commit -m "Migrate from SQLite to Neo4j for knowledge graph implementation"

Write-Host ""
Write-Host "Branch created successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Connect to your GitHub repository:"
Write-Host "   git remote add origin https://github.com/yourusername/your-repo.git" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Push the branch to GitHub:"
Write-Host "   git push -u origin $BRANCH_NAME" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Create a pull request on GitHub to merge the Neo4j changes" -ForegroundColor Cyan 