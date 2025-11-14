
# AGENTS.md (Grokking / RL Research Repo)

**Purpose**

Define how an automated coding agent plans, edits, runs, and validates code in this Python grokking / RL research repository so changes are safe, reproducible, and aligned with project goals.

---

## 1. Goals and Scope

Goals

- Improve experiment velocity without breaking `main`.
- Keep a clean, linear history with a rebase-first workflow.
- Prefer PyTorch defaults and minimize external dependencies.
- Produce reproducible experiments, metrics, and plots for grokking studies.
- Leave scaffolding for optional backends without blocking default runs.

Out of scope

- Publishing packages.
- Modifying secrets or CI credentials.
- Long-running expensive cloud jobs without explicit user request.

---

## 2. Agent Roles

- **Planner**: Turn a user request into a concrete plan and a small diff set.
- **Implementer**: Write code with tests, typing, and docstrings.
- **Reviewer**: Self-review and propose alternatives or rollbacks.
- **Runner**: Execute fast checks locally and never run destructive steps without opt-in.

Each step leaves an artifact in the PR or commit message.

---

## 3. Decision Policy (Impact, Risk, Cost)

Impact

- Low: comments, docs, config toggles.
- Medium: new small module or function, minor refactors.
- High: API changes, train loop edits, data schema changes.

Risk

- Low: local lints, added tests, non-executable docs.
- Medium: isolated module change with tests.
- High: touching training loop, logging schema, or configs used in CI.

Cost

- Low: < 30 seconds unit tests and static checks.
- Medium: quick CPU-only script runs.
- High: GPU training or large dataset downloads.

Rule

- Only proceed when Impact + Risk + Cost fits the user instruction and guardrails.
- Default to the smallest diff that satisfies the requirement.
- For high-risk changes, split into staged PRs: interface first, behavior second, performance third.

---

## 4. Repository Guardrails

- Python 3.11.
- `requirements.txt` is the single source of dependency truth.
- PyTorch is the default backend.
- Alternate backends must be optional and disabled by default.
- Use rebase-first Git workflow. No merge commits on feature branches.
- All changes must pass `make check` (ruff + black + mypy + pytest + coverage).
- Use `make smoke` whenever training/eval code paths change.

---

## 5. Standard Working Branch Flow

1. Create a small feature branch

```bash
git switch -c feat/<short-task-name>
```

2. Make minimal changes with tight commits

```bash
git add -p
git commit -m "feat: <concise change>"
```

3. Rebase often

```bash
git fetch origin
git pull --rebase origin main
```

4. Push when checks pass

```bash
git push -u origin HEAD
```

---

## 6. Coding Standard

- Typing: mypy-clean public APIs, explicit return types.
- Style: black, ruff.
- Tests: pytest unit tests for new functions and bug fixes.
- Docs: docstring for every public function.
- Config: do not break existing configs. Add new ones as opt-in.

---

## 7. Safe-Run Protocol

Default quick checks:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
make check
```

Experiment smoke test

- CPU-only tiny run with fixed seed.
- Writes logs to a temp run directory.
- Generates tiny plots to verify pipeline wiring.

Never do by default

- Long GPU training.
- Network downloads > few MB.
- Any destructive operation.
- Modifying CI secrets or repo settings.

---

## 8. Config and Backend Guidance

- PyTorch is first-class.
- Alternate backends: leave disabled by default with feature flags.
- CSV schema changes require synchronized updates to readers, writers, and tests.
- Spectral or Laplacian features: minimal vetted version or disabled hook.

---

## 9. Logging and Metrics

Minimum logs per run

- Train loss.
- Test loss.
- Train accuracy.
- Test accuracy.
- Optional: weight norm, spectral energy.

Plotting rules

- Linear and log x-axis supported.
- Grokking step marker: first step staying above threshold.
- Plotting code must accept single or list of run dirs.

---

## 10. PR Checklist

- [ ] Focused branch and diff.
- [ ] Code compiles/imports.
- [ ] Unit tests added/updated.
- [ ] ruff, black, mypy pass.
- [ ] pytest passes.
- [ ] make smoke runs when needed.
- [ ] README/docs updated.
- [ ] Changelog entry if needed.

---

## 11. Run Permission Matrix

| Action type                         | Default | Needs explicit user ok |
| ----------------------------------- | ------- | ---------------------- |
| Lint, type check, unit tests        | Allowed | No                     |
| Edit docs, comments                 | Allowed | No                     |
| Add small pure-Python helper        | Allowed | No                     |
| Modify train loop or logging schema | Ask     | Yes                    |
| Download datasets > 10 MB           | Ask     | Yes                    |
| Long training > 2 minutes           | Ask     | Yes                    |
| Change CI config                    | Ask     | Yes                    |

---

## 12. Prompt Protocol

Planning prompt

- Summarize goal in 1 sentence.
- List 3–5 minimal steps.
- Identify schema/API changes.
- State checks you will run.

Implementation prompt

- Minimal diff with signatures, tests, docstrings.
- Config flags and defaults.
- Run commands.

Review prompt

- Self-critique.
- Verify checklist.
- Rollback plan.

---

## 13. Failure Handling & Rollback

- If a check fails, post exact command and error.
- Propose hotfix or revert.
- Never push failing main.

---

## 14. Communication Style

- Concise. Commands always listed.
- Reference paths.
- Prefer small diffs.

---

## 15. Example: Add Grokking Step Marker

Plan

1. Parse metrics CSV.
2. Compute threshold-sustained accuracy step.
3. Plot with optional vertical marker.
4. Add unit test.

API sketch:

```python
def find_grokking_step(steps, test_acc, threshold=0.90, patience=50) -> int | None:
    ...
```

