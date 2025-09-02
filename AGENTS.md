# Repository Guidelines

## Project Structure & Module Organization
- Core work lives in `notebooks/` (e.g., `change_point.ipynb`, `kalman_rayleigh.ipynb`).
- When code stabilizes, promote reusable logic into `src/hmm_project/` (create as needed) and import from notebooks.
- Keep data out of Git. Use `data/raw/` and `data/processed/` (both gitignored) for local inputs/outputs.
- Place tests in `tests/` mirroring `src/` modules (add when `src/` exists).

## Build, Test, and Development Commands
- Create environment: `python -m venv .venv && source .venv/bin/activate`.
- Install deps (if present): `pip install -r requirements.txt`.
- Launch notebooks: `jupyter lab` or `jupyter notebook`.
- Execute a notebook headless: `jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb --output /tmp/out.ipynb`.
- Run tests (when added): `pytest -q` (consider `pytest --maxfail=1 -q`).

## Coding Style & Naming Conventions
- Language: Python 3.10+ preferred for notebooks and `src/`.
- Formatting: Black (line length 88), isort (profile "black"). Lint with Ruff or Flake8.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Notebooks: `NN_topic.ipynb` (e.g., `01_hmm_intro.ipynb`) or `YYYY-MM-DD_topic.ipynb` for dated experiments.
- Move non-trivial functions from notebooks into `src/hmm_project/` and import them to keep notebooks lightweight.

## Testing Guidelines
- Framework: Pytest. Place files as `tests/test_<module>.py`.
- Coverage: target ≥80% for `src/` modules; add `pytest-cov` if useful.
- Notebooks: add smoke tests via `nbval` or execute critical notebooks with `nbconvert` in CI to catch regressions.

## Commit & Pull Request Guidelines
- Commits: follow Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Example: `feat(hmm): add viterbi path decoder`.
- PRs: small, focused, with clear description, linked issues, and before/after artifacts (plots, metrics). Remove/strip notebook outputs before committing (use `nbstripout` or pre-commit hooks).
- Checks: ensure notebooks run top-to-bottom; tests pass; linters/formatters clean.

## Security & Data Tips
- Never commit credentials or large datasets. Use `.env` for secrets and add to `.gitignore`.
- Prefer deterministic seeds in experiments; record environment (Python, package versions) in the PR.
