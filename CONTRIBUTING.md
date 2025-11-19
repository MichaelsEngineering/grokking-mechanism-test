# Contributing to This Project

We're excited you're here to contribute! To help maintain a clean, high-quality, and easy-to-manage codebase, we follow a specific Git workflow. Please read these guidelines before you start.

## 🤝 Core Philosophy

Our goal is to maintain a **linear, clean, and readable Git history**. We achieve this by:
* Using a **rebase-centric workflow** (we avoid merge commits on feature branches).
* Working in **small, focused, well-named feature branches**.
* Using **pre-commit hooks and formatters** to ensure consistent code style.
* Protecting the `main` branch to ensure stability.

---

## 🛠️ One-Time Local Setup

Before your first contribution, please configure your local environment to match our workflow.

### 1. Configure Git

Run these commands in your terminal to set global defaults. This ensures your local Git operations align with our rebase strategy and avoid common errors.
```bash
### Safer rebases and easier conflict recovery
git config --global rerere.enabled true
git config --global rebase.updateRefs true
git config --global rebase.missingCommitsCheck warn

### Keep your local tree tidy
git config --global fetch.prune true

### Nudge contributors toward tagging releases
git config --global push.followTags true

### Consistent line endings across OSes
git config --global core.autocrlf input

### Useful quality-of-life
git config --global init.defaultBranch main
git config --global commit.verbose true
git config --global push.autoSetupRemote true

---

## 🐞 Reporting Bugs and Proposing Features

We welcome bug reports and feature requests! To ensure we have all the necessary information, please use our combined **[Bug Report / Feature Request](/.github/ISSUE_TEMPLATE/bug_feature_report.md)** template when creating a new issue.

This template is designed to capture all the relevant details for both bugs and features in a structured way, which helps us understand the context and aids our AI-assisted development workflow. When you create an issue, you will be prompted to use this template.
```
