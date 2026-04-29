# PSINS Round Record Template

> 用于每一轮实验开始前和结束后强制记录，避免“试完再想自己在干什么”。

---

## A. Round 基本信息

- Round name:
- Round type: `mainline refine` / `repair branch` / `new mechanism probe` / `regime branch`
- Base candidate:
- Dataset / regime:
- Seed:

## B. 本轮目标

- Primary goal:
- Secondary goal:
- This round is NOT trying to do:

## C. Allowed knobs

- knob group 1:
- knob group 2:

## D. Protected metrics

- must hold:
- can tolerate tiny regression:
- absolutely cannot regress:

## E. Candidate design

### candidate 1
- name:
- changed knobs:
- rationale:
- expected benefit:
- possible risk:

### candidate 2
- name:
- changed knobs:
- rationale:
- expected benefit:
- possible risk:

### candidate 3
- name:
- changed knobs:
- rationale:
- expected benefit:
- possible risk:

## F. Scoring / gate

- clean win definition:
- partial signal definition:
- no useful signal definition:
- formalize gate:

---

## G. Result summary

- winner:
- result class: `clean win` / `partial signal` / `no useful signal` / `negative lesson`
- one-line conclusion:

## H. Metric deltas vs base

- key improves:
- key regressions:
- overall mean:
- overall median:
- overall max:

## I. Mechanism learning

- what probably worked:
- what probably did not work:
- is this gain structural or just redistribution?

## J. Next experiment generation

- keep:
- remove:
- next best repair direction:
- next best new-mechanism direction:
- should formalize now? yes / no

## K. Artifacts

- candidate_json:
- summary_json:
- report_md:
- formal_method_file:
- formal_result_json:
