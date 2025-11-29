# GitHub Setup Guide

This guide expands on the probability-lab workflow: combine Git, GitHub CLI, and GitLens to keep the project reproducible and collaborative.

## Initialize the Repository
1. `git init`
2. `git add .`
3. `git commit -m "feat: bootstrap llm-lab"`

## Create the GitHub Repository
- **Using GitHub CLI**
  ```bash
  gh repo create USERNAME/llm-lab --private --source=. --remote=origin --push
  ```
- **Using Web UI**
  1. Log in to GitHub and click **New Repository**.
  2. Name it `llm-lab` and choose visibility (public/private).
  3. Skip initializing with README (already present).
  4. Copy the remote URL and run:
     ```bash
     git remote add origin git@github.com:USERNAME/llm-lab.git
     git push -u origin main
     ```

## GitLens Setup
- Install the GitLens extension in VS Code.
- Open the repository folder (`code .`).
- Enable the **Focus Mode** view to inspect diffs and blame annotations.
- Configure commit message template via GitLens (`Settings → GitLens → Commit Message`).

## Recommended Workflows
- **Feature branches**
  ```bash
  git checkout -b feat/new-experiment
  # edit code
  git commit -am "feat: add new experiment scaffold"
  git push -u origin feat/new-experiment
  ```
- **Pull Request checklist**
  - Run `pre-commit run --all-files`
  - Run `pytest`
  - Ensure `docs/` updates accompany code changes when interfaces change.
- **Release tags**
  ```bash
  git tag -a v0.1.0 -m "First public release"
  git push origin v0.1.0
  ```

## Platform Notes
- **macOS**: Use the system `ssh-agent` (`ssh-add --apple-use-keychain ~/.ssh/id_ed25519`).
- **Linux**: Configure `~/.ssh/config` with `Host github.com` to store identity settings.
- **Windows**: Use the Git Credential Manager (ships with recent Git for Windows) or generate keys via `ssh-keygen` and load them in Pageant.

## Troubleshooting
- **`gh` fails with auth errors**: run `gh auth logout` followed by `gh auth login`.
- **SSH key denied**: ensure the public key is added to GitHub and `chmod 600 ~/.ssh/id_ed25519`.
- **Diverging branches**: run `git fetch origin main` and `git rebase origin/main` before pushing.
- **Large files blocked**: consider Git LFS or store artifacts outside the repository; `.gitignore` already excludes checkpoints and logs.
