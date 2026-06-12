# Roadmap

gap v1 draws a deliberate line: **language → verified pick-and-place graphs
on LIBERO and real Franka/UR hardware.** Everything inside that line is
shipped, tested, and gated; everything outside it was cut for scope — named
here, not hidden — and is planned to return after v1.

## What v1 is

- Natural-language instructions compiled into typed, validated workflow
  graphs ([Generating graphs](../authoring/generation.md)), plus the same
  artifact authored by hand ([The builder API](../authoring/builder.md)).
- Execution on LIBERO simulation and on real Franka (full manipulation) and
  UR+ZED (perception-only) connectors
  ([Connectors](../real-robots/connectors.md)).
- Verification against simulator ground truth via LLM-authored checkpoints
  ([Checkpoints](../running/checkpoints.md)).
- A release gate that holds the line: the grocery-fulfillment acceptance
  benchmark must clear **≥90% success** before any release
  ([Benchmarking](../benchmarks/benchmarking.md)).
- Geometric grasping: oriented-bounding-box top-down candidates executed
  via CuRobo planning or direct IK.

## Deliberately after v1

These were cut for scope, not for lack of interest — several existed in the
research codebase gap was ported from and return once the v1 surface is
stable:

- **Execution-feedback graph repair.** Today the script-fix loop repairs
  *validation* errors at generation time; a refine loop that feeds
  *execution* failures back into the agent to repair the graph is post-v1.
- **Learned grasp planners (graspgen / m2t2).** v1 ships geometric grasping
  only. Learned grasp generation was prototyped pre-release (graspgen ran
  with a shipped example; m2t2 and graspnet were implemented) and returns
  as tool bundles after v1.
- **A remote model-serving tier.** Everything runs in-process by design;
  the `[ray]` extra lets benchmark workers share model actors, but a
  first-class remote serving tier for heavy models on another machine is
  post-v1.
- **Bimanual manipulation.** The connector tools already take `arm_id=0`
  (multi-arm-ready), but two-arm variants and bimanual skills are cut.
- **Isaac-based rehearsal.** An Isaac-based rehearsal pipeline existed in
  the research codebase and is cut from v1; v1 verification is checkpoint
  evaluation in the execution sim itself
  ([Checkpoints](../running/checkpoints.md)).
- **Non-pick-place domains.** Articulated manipulation (e.g. turning a
  stove knob), contact-rich insertion (USB), and long-horizon multi-stage
  tasks were prototyped pre-release and are out of v1; the skill catalog
  ships pick-and-place strategies.
- **Self-hosted pointing VLMs.** The `molmo` tool bundle (self-hosted via
  vLLM) is in the registry but optional; the supported v1 path for
  detection and spatial reasoning is the API-based `gemini-er` bundle. See
  the [tool catalog](../skills/tool-catalog.md).

## How to influence it

The cheapest way to extend gap is usually not the engine: new perception,
grasping, or transport strategies are skill bundles — one directory, one PR
to [open-robot-skills](gh-skills:.)
([Authoring bundles](../skills/authoring-bundles.md)). For engine work, see
[Contributing](contributing.md); if you depend on one of the cuts above,
open an issue so post-v1 priorities reflect real demand.
