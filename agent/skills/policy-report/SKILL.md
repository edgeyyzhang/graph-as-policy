---
name: policy-report
description: Make the policy card for a GaP policy graph and publish it as a Claude artifact page. Samples N layouts of the LIBERO task, evaluates the graph on them with video, builds one self-contained report.html (task, graph drawing, a two-panel video per trajectory, per-layout evaluation with failure modes, parameters per node) and publishes it. Use when the owner asks for a report on a generated or rehearsed policy graph.
---

# Policy report

Usage: `/policy-report GRAPH_DIR SUITE/TASK [--trajs N] [--seed S] [--skip-eval]`

The report is about the policy, not about how it was made. Four fixed sections:
0 task, 1 policy (graph drawing and a representative video), 2 evaluation of
the sampled layouts with one video each and the failure modes, 3 parameters of
every node. The generator in `scripts/` is deterministic; do not write prose
into the report yourself.

## Steps

1. Resolve `GRAPH_DIR`. For a finished rehearsal loop it is
   `gap_loops/<loop>/results/graph`; for a plain generation it is the output
   directory holding `workflow.json`. Call the report `<name>` after the loop or
   graph directory.
2. Pick an idle GPU with `nvidia-smi`.
3. Run the driver from the graph-as-policy checkout (in this project
   `./graph-as-policy`; this skill lives inside it at `agent/skills/policy-report`):
   ```bash
   CUDA_VISIBLE_DEVICES=<n> agent/skills/policy-report/scripts/policy_report.sh \
       GRAPH_DIR SUITE/TASK outputs/policy_report/<name> --trajs <N> [--seed <S>]
   ```
   `--trajs N` draws N layouts at random from the task's 16 (seeded, recorded in
   `eval/sample.json`) and evaluates only those, each with a video; default 4.
   `--trajs all` evaluates every layout with videos only for the failures. About
   75 seconds per layout: run it in the background and do other work meanwhile.
   If `outputs/policy_report/<name>/eval` exists, add `--skip-eval` to rebuild
   only the page.
4. Read the driver's last line: path, size in MB, successes, and OVER BUDGET if
   the page exceeds the artifact limit. If over, rerun with
   `--skip-eval -- --max-mb 14 --speed 4`.
5. Publish `outputs/policy_report/<name>/report/report.html` with the Artifact
   tool: `icon: "report"` and a one-sentence `description` naming the task, the
   drawn cases and the score. The file is a complete page fragment (title,
   styles, both themes, embedded media). On a rebuild, republish to the same URL.
6. Reply with the link, the score over the evaluated layouts naming the drawn
   cases, and each failure with the node it ended in, as section 2 lists them.
   Nothing else.

## Files

| File | Role |
|---|---|
| `scripts/policy_report.sh` | Driver: samples the layouts, runs `gap rehearse` with frames and video, then the generator |
| `scripts/policy_card.py` | Generator: reads the evaluation and the graph, writes `report.html` and `report.json` |

The driver uses `examples/rehearse_loop/run.sh` for the rehearsal and
`examples/rehearse_loop/compose_video.py` for the two-panel videos.

## Notes

- `report.json` beside the page holds every number and the parameter table; use
  it for a table across tasks instead of parsing HTML.
- A failure whose graph exit is `success` means the policy believed it was done
  and the simulator disagreed; the generator says so in the page.
- Install for Claude Code by linking or copying this directory into the
  project's `.claude/skills/`.
