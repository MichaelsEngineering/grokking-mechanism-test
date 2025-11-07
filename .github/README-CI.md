Detected toolchain: Pandoc workflow (PDF_MAIN=main.md; ensure this file exists before running CI)

## Paper CI Checks
- `build-pdf` builds `main.pdf` using Pandoc and xelatex, then uploads it as the `paper-pdf` artifact.
- `spellcheck` runs codespell (ignoring `vendor`, `build`, PDFs, and `.git`) to keep prose tidy.
- `refs-and-citations` enforces citation/bibliography checks when the LaTeX toolchain is active; non-LaTeX toolchains log the detected mode and exit successfully.
- `check-large-files` verifies the full Git history and fails when blobs larger than 10 MB are committed outside Git LFS (logs tracked LFS files for visibility).

## Branch Ruleset Guidance
- Apply a new branch ruleset named `Paper Protected` targeting `main`, `release/*`, `camera-ready/*`, and `exp/*`.
- Mark the four CI jobs above as **Required** checks within the ruleset once they appear in GitHub’s picker.
- Require pull requests for every change; approvals are optional for a solo maintainer, but keep review enabled for future collaborators.
- Prefer linear history (rebase or squash) to keep the protected branches tidy; signed commits remain optional.
