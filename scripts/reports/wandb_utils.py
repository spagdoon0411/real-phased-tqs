"""
Shared helpers for pulling run data out of Weights & Biases for the report scripts in
this directory. Requires `wandb login` to have been run once on this machine.
"""

import sys

import wandb
from wandb.apis.public import Run

DEFAULT_PROJECT = "real-phased-tqs"


def resolve_run(api: wandb.Api, run: str | None, project: str, entity: str | None) -> Run:
    """
    Resolves a run identifier to a `wandb.apis.public.Run`.

    `run` may be:
      - None, in which case the most recently created run in the project is used.
      - a full path "entity/project/run_id".
      - a bare run id or display name (e.g. "decent-lake-4"), looked up within
        `entity`/`project` (falling back to the caller's default entity if `entity`
        is None).
    """
    entity = entity or api.default_entity
    if entity is None:
        raise SystemExit("No W&B entity found; pass --entity or run `wandb login`.")
    project_path = f"{entity}/{project}"

    if run is not None and run.count("/") == 2:
        return api.run(run)

    if run is not None:
        try:
            return api.run(f"{project_path}/{run}")
        except wandb.errors.CommError:
            pass
        matches = list(api.runs(project_path, filters={"display_name": run}, order="-created_at"))
        if not matches:
            raise SystemExit(f"No run named or IDed '{run}' found in project '{project_path}'.")
        if len(matches) > 1:
            print(
                f"Multiple runs named '{run}' in '{project_path}'; using the most recent ({matches[0].id}).",
                file=sys.stderr,
            )
        return matches[0]

    matches = list(api.runs(project_path, order="-created_at"))
    if not matches:
        raise SystemExit(f"No runs found in project '{project_path}'.")
    return matches[0]


def fetch_history(run: Run) -> list[dict]:
    """
    Returns every logged row for `run` as a list of dicts (unsampled, unlike
    `Run.history()`'s default 500-row cap).
    """
    return list(run.scan_history())


def extract_series(rows: list[dict], key: str) -> tuple[list[float], list[float]]:
    """
    Pulls (step, value) pairs out of `rows` for `key`, skipping rows where the metric
    wasn't logged (e.g. `sym_loss` before symmetrization was enabled).
    """
    steps, values = [], []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        steps.append(row.get("_step"))
        values.append(value)
    return steps, values
