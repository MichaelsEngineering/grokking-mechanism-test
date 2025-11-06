# AGENT.md

**Purpose**
Define how an automated coding agent plans, edits, runs, and validates code in this repository so that changes are safe, reviewable, and aligned with project goals.

## 1) Goals and Scope

* Improve developer velocity without breaking main.
* Keep a clean, linear history with rebase-first workflow.
* Prefer PyTorch defaults and minimize external dependencies.
* Produce reproducible experiments and plots for grokking studies.
* Leave scaffolding for optional backends without blocking default runs.

Out of scope

* Publishing packages.
* Modifying secrets or CI credentials.
* Long-running expensive cloud jobs without an explicit user request.

---

## 2) Agent Roles

* **Planner**: turns a user request into a concrete plan and a small diff set.
* **Implementer**: writes code with tests, typing, and docstrings.
* **Reviewer**: self-reviews and proposes alternatives or rollbacks.
* **Runner**: executes fast checks locally, never runs destructive steps without opt-in.

Each step leaves an artifact in the PR or commit message.

---

## 3) Decision Policy

The agent scores each proposed action before doing it.

**Impact**
Low: comments, docs, config toggles
Medium: new small module or function, minor refactors
High: API changes, train loop edits, data schema changes

**Risk**
Low: local lints, added tests, non-executable docs
Medium: isolated module change with tests
High: touching training loop, logging schema, or configs used in CI

**Cost**
Low: < 30 seconds unit tests and static checks
Medium: quick CPU-only script runs
High: GPU training or large dataset downloads

**Rule**

* Only proceed when Impact + Risk + Cost fits the current user instruction and the repository guardrails below.
* Default to the smallest diff that satisfies the requirement.
* If a high-risk change is requested, split into staged PRs: interface first, behavior second, performance third.

---

## 4) Repository Guardrails

* Python 3.11.
* `requirements.txt` is the single source of dependency truth.
* PyTorch is the default backend. Keep alternate-backend hooks optional and disabled by default.
* Use rebase-first Git workflow. No merge commits on feature branches.
* All changes must pass `make check` (ruff + black + mypy + pytest + coverage). Use `make smoke` whenever training/eval code paths change.

---

## 5) Standard Working Branch Flow

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

## 6) Coding Standard

* **Typing**: mypy-clean public APIs, explicit return types.
* **Style**: black, ruff.
* **Tests**: pytest unit tests for new functions and any bug fixes.
* **Docs**: docstring for every public function, include simple usage.
* **Config**: do not break existing configs. Add new ones as opt-in.

---

## 7) Safe-Run Protocol

The agent runs only fast checks by default.

**Local quick checks**

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
make check
```

**Experiment smoke test**

* CPU-only short run with tiny epochs and a seed.
* Writes logs to a temp run directory.
* Generates plots only from tiny logs to confirm pipeline wiring.

**Never do by default**

* Long GPU training.
* Network downloads larger than a few MB.
* Any operation that overwrites user data.
* Modifying CI secrets or repo settings.

---

## 8) Config and Backend Guidance

* PyTorch paths are first-class.
* If a requested feature touches alternate backends, add a feature flag in config and leave it disabled by default.
* If the CSV schema changes, update readers, writers, and tests together.
* For spectral or Laplacian features that are not implemented, either disable the hook in config or implement the minimal vetted version with tests.

---

## 9) Logging and Metrics

Minimum logs per run

* Train loss
* Test loss
* Train accuracy
* Test accuracy
* Optional: weight norm, spectral energy placeholder

Plotting rules

* X-axis supports linear and log scale.
* Grokking step marker: first step where test accuracy stays ≥ threshold.
* All plotting code must accept a single run dir or a list of run dirs.

---

## 10) PR Checklist (used by the agent)

* [ ] Small, focused branch name and diff.
* [ ] Code compiles and imports cleanly.
* [ ] New or changed functions have unit tests.
* [ ] ruff, black, mypy pass.
* [ ] pytest passes locally.
* [ ] `make smoke` runs locally (when training/eval behavior changes) and writes a small log.
* [ ] README or inline docs updated if behavior changed.
* [ ] Changelog note in PR body.

---

## 11) Run Permission Matrix

| Action type                         | Default | Needs explicit user ok |
| ----------------------------------- | ------- | ---------------------- |
| Lint, type check, unit tests        | Allowed | No                     |
| Edit docs, comments                 | Allowed | No                     |
| Add a small pure-Python helper      | Allowed | No                     |
| Modify train loop or logging schema | Ask     | Yes                    |
| Download datasets > 10 MB           | Ask     | Yes                    |
| Long training > 2 minutes           | Ask     | Yes                    |
| Change CI config                    | Ask     | Yes                    |

---

## 12) Prompt Protocol for the Agent

**Planning prompt**

* Summarize the user goal in 1 sentence.
* List 3 to 5 smallest viable steps.
* Identify any schema or public API change.
* State the quick checks you will run.

**Implementation prompt**

* Propose a minimal diff, show function signatures, tests, and docstrings.
* Note any config flag default and why.
* Provide run commands for checks and a smoke test.

**Review prompt**

* Self-critique risks and alternatives.
* Confirm all items in the PR checklist.
* Provide a rollback plan.

---

## 13) Example: Disable spectral.compute by default

Plan

1. Add `spectral: { enabled: false }` in default config.
2. Guard all spectral calls with a feature flag.
3. Add a unit test that spectral disabled runs do not write spectral CSV columns.
4. Update README config table.

Diff sketch

```python
# config_defaults.py
defaults = {
    "spectral": {"enabled": False},
    # ...
}

# train.py
if cfg["spectral"]["enabled"]:
    spectral_energy = compute_spectral_energy(model, batch)
    logger.log_scalar("spectral_energy", spectral_energy, step)
```

Test sketch

```python
def test_runs_without_spectral(tmp_path):
    cfg = load_cfg()
    cfg["spectral"]["enabled"] = False
    run_dir = run_quick_smoke(cfg, out_dir=tmp_path)
    assert not (run_dir / "spectral.csv").exists()
```

---

## 14) Example: Add grokking step marker to visualize.py

Plan

1. Parse metrics CSV.
2. Compute first test-accuracy step that stays above threshold.
3. Plot loss and accuracy with optional vertical marker.
4. Add a unit test for the marker function.

API

```python
def find_grokking_step(steps, test_acc, threshold=0.90, patience=50) -> int | None:
    ...
```

---

## 15) Failure Handling and Rollback

* If a check fails, the agent posts the failure summary with the exact command and traceback snippet.
* The agent proposes either a 1-line hotfix or a revert commit.
* Never push failing main. Keep failures on the feature branch.

---

## 16) Communication Style

* Keep messages concise, list the exact commands to reproduce.
* Prefer small PRs. Include a short rationale and links to code lines.
* Avoid em dashes. Use simple punctuation.

---

## 17) Trust but Verify

* The agent must assume responsibility for passing checks before asking for review.
* Human reviewers remain the final gate for medium and high impact changes.

---

If desired, add a `make check` target for the agent’s local validation:

```makefile
check:
	python -m pip install -r requirements.txt
	ruff check .
	black --check .
	mypy .
	pytest -q
```
