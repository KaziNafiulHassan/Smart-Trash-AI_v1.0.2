#!/bin/bash

# Create GitHub Branch Script for Neo4j Migration

echo "=== Smart Trash AI - GitHub Branch Creator ==="
echo "This script will help you create and push a branch for the Neo4j migration."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: Git is not installed. Please install git first."
    exit 1
fi

# Check if current directory is a git repository
if [ ! -d .git ]; then
    echo "Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Create and switch to a new branch
BRANCH_NAME="neo4j-migration"
echo "Creating and switching to branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

# Add all files
git add .

# Commit changes
git commit -m "Migrate from SQLite to Neo4j for knowledge graph implementation"

echo ""
echo "Branch created successfully!"
echo ""
echo "Next steps:"
echo "1. Connect to your GitHub repository:"
echo "   git remote add origin https://github.com/yourusername/your-repo.git"
echo ""
echo "2. Push the branch to GitHub:"
echo "   git push -u origin $BRANCH_NAME"
echo ""
echo "3. Create a pull request on GitHub to merge the Neo4j changes" 