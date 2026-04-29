---
name: psins-benchmark
description: |
  Benchmark and refactor mixed PSINS/INS calibration experiment codebases. Use when the user gives a folder with many calibration/filtering experiment scripts and wants you to: (1) identify methods hidden inside large scripts, (2) extract each method into separate runnable files, (3) build a shared benchmark harness, (4) run methods and compare calibration accuracy / convergence speed / stability, or (5) summarize the methods into Feishu docs.
---

# PSINS Benchmark Skill

Use this skill when a research code folder mixes multiple calibration methods inside a few large scripts and the user wants a reusable benchmark workflow.

## Default workflow

1. **Scan the folder**
   - Identify core modules, test scripts, output folders, and duplicated experiment branches.

2. **Identify hidden methods inside each script**
   - Do not stop at file-level grouping.
   - Detect if one script contains multiple experimental methods (e.g. clean baseline, noisy baseline, SCD).

3. **Extract methods**
   - Create separate method files for each actual method.
   - Also extract:
     - shared dataset / config setup
     - result extraction
     - plotting

4. **Build benchmark structure**
   - Use a structure like:
     - `methods/`
     - `configs/`
     - `results/`
     - `summary/`

5. **Run benchmark**
   - Prefer local `exec` for long-running code.
   - Avoid using subagents as the main runner for CPU / Python experiment execution unless isolation is truly needed.
   - Use longer timeouts for actual benchmark runs.

6. **Summarize results**
   - Produce at least:
     - run status
     - runtime
     - result structure
     - calibration accuracy summary
     - convergence speed summary
     - stability / side-effect analysis

## Important rules

- **Do not confuse file grouping with method extraction.**
  If a single script runs 2–3 methods, extract all 2–3 methods.

- **Do not over-compress results.**
  Keep enough detail for the user to compare methods fairly.

- **Use Feishu docs for summaries when helpful.**
  But keep technical benchmark artifacts on disk as files first.

## References

Read this when needed:
- `references/workflow.md`
