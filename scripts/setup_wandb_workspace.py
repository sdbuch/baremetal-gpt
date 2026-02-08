"""Generate a wandb workspace view with overlay panels for per-layer metric comparison.

Pulls metric keys from the most recent run (or a specified run) in a project,
then creates a saved view with one LinePlot per unique base metric, overlaying
all layers on the same chart.
"""

import os
import re

import tyro
import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

LAYER_RE = re.compile(r"layer_\d+\.")


def get_metric_keys(entity: str, project: str, run_id: str) -> list[str]:
  api = wandb.Api()
  run = api.run(f"{entity}/{project}/{run_id}")
  print(f"Using run: {run.id} ({run.name})")
  return [k for k in run.summary.keys() if not k.startswith("_")]


def build_sections(keys: list[str]) -> list[ws.Section]:
  by_prefix: dict[str, list[str]] = {}
  for key in keys:
    if "/" in key:
      prefix = key.split("/", 1)[0]
      by_prefix.setdefault(prefix, []).append(key)

  sections = []

  # Train scalars: individual panels
  if "train" in by_prefix:
    panels = [
      wr.LinePlot(title=k.split("/", 1)[1], y=[k]) for k in sorted(by_prefix["train"])
    ]
    sections.append(ws.Section(name="train", panels=panels, is_open=True))

  # Per-layer overlay sections
  for prefix in ("fwd", "weight", "grad", "update"):
    if prefix not in by_prefix:
      continue

    layer_bases: dict[str, None] = {}  # insertion-ordered set
    singleton_keys: list[str] = []
    for key in by_prefix[prefix]:
      suffix = key.split("/", 1)[1]
      if LAYER_RE.match(suffix):
        base = LAYER_RE.sub("", suffix)
        layer_bases[base] = None
      else:
        singleton_keys.append(key)

    panels: list[wr.LinePlot] = []
    for base in sorted(layer_bases):
      panels.append(
        wr.LinePlot(
          title=base,
          metric_regex=rf"{prefix}/layer_\d+\.{re.escape(base)}",
        )
      )
    for key in sorted(singleton_keys):
      panels.append(wr.LinePlot(title=key.split("/", 1)[1], y=[key]))

    sections.append(ws.Section(name=prefix, panels=panels, is_open=False))

  return sections


def main(
  project: str,
  run_id: str,
  entity: str = os.environ.get("WANDB_ENTITY", ""),
  name: str = "Layer Overlays",
):
  """Generate a wandb workspace with per-layer overlay panels.

  Args:
    project: wandb project name
    run_id: run ID to pull metric keys from (must have aux metrics logged)
    entity: wandb entity (user or team), defaults to $WANDB_ENTITY
    name: saved view name
  """
  if not entity:
    raise ValueError("entity must be provided via --entity or $WANDB_ENTITY")
  keys = get_metric_keys(entity, project, run_id)
  sections = build_sections(keys)
  if not any(s.name != "train" for s in sections):
    raise RuntimeError(
      "No overlay sections generated — the selected run has no fwd/weight/grad/update metrics."
    )

  workspace = ws.Workspace(
    entity=entity,
    project=project,
    name=name,
    sections=sections,
  )
  workspace.save()
  print(f"Saved workspace view: {workspace.url}")


if __name__ == "__main__":
  tyro.cli(main)
