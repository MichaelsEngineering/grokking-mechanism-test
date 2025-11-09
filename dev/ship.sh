#!/usr/bin/env bash
set -euo pipefail

# Usage: dev/ship.sh "feat: add parity experiment config"
MSG="${1:-}"

# Ensure clean worktree
if ! git diff --quiet; then
  echo "Staging changes..."
  git add -A
fi

# Create feature branch if on main
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CUR_BRANCH" = "main" ]; then
  NEW="feat/$(date +%y%m%d-%H%M%S)"
  echo "On main. Creating branch $NEW"
  git checkout -b "$NEW"
fi

echo "Running checks..."
make check

if [ -z "$MSG" ]; then
  echo "No message provided. Opening Conventional Commit prompt..."
  cz commit
else
  git commit -m "$MSG"
fi

echo "Rebasing on latest origin/main..."
git fetch origin
git rebase origin/main

echo "Pushing..."
git push -u origin HEAD

REPO_PATH=$(git config --get remote.origin.url | sed -E 's#(git@|https://)github.com[:/]|\.git##g')
BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Open a PR:"
echo "https://github.com/${REPO_PATH}/compare/main...${BRANCH}?expand=1"
