# CognitiveTrack Agent Instructions

**Last updated:** 2026-08-18
**Current phase:** Both formal v6.4 SFT releases and their processor preflights are complete;
mixed-package training and online inference alignment are in progress.

Read this file with [`docs/README.md`](docs/README.md),
[`docs/project_status.md`](docs/project_status.md), and [`docs/setup.md`](docs/setup.md).
Do not ask the user to restate facts recorded there.

## Current protocol

CognitiveTrack studies pure-VLM long-term single-object tracking under:

```text
Sequence -> initialize -> track -> ResultWriter -> Evaluator
```

The student is `Qwen3-VL-4B-Instruct`. Native Prompt version is `6.4.0` with three images:

1. Image 1: an earlier full present frame with a red target box; identity anchor.
2. Image 2: three boxed trusted observations, chronological left-to-right with white separators.
3. Image 3: current full search frame without a box.

The Qwen3-VL response is:

```json
{"bbox_2d":[100,120,400,520],"status":"present","memory_update":null}
```

`bbox_2d` is Image-3 `[0,1000] xyxy`. A non-null `memory_update` is a complete replacement
dynamic referring expression, never an appended delta. Permanent identity and dynamic history must
be gated so a bad prediction cannot overwrite the identity anchor. Dynamic memory may explicitly
describe disappearance and reappearance; the permanent visual identity remains separate.

## Data taxonomy and supervision

Use user-facing/data names `tracking_sft` and `state_update_sft`; do not introduce new “Core” or
“Memory stage” names. Historical `tracking_core` remains a read-only compatibility alias.

`tracking_sft` covers LaSOT, TNL2K, and MGIT tiny/train with 27 legal visual combinations:

- temporal event: continuous present, absent, reappearance;
- history quality: clean, one jittered box, one stale box;
- completeness: H0-H3, with H0 clean only, H1 no stale, stale requiring H2/H3.

`tracking_sft` does not provide memory labels: placeholder `memory_update:null` is `masked_unknown`
for both present and absent rows, and only its value is masked. Bbox, status, field names, JSON
closure, and suffix remain supervised.

`state_update_sft` combines:

1. all reliable MGIT official action-segment labels;
2. 2,329 additional LaSOT/TNL2K labels produced by the remote `qwen3-vl-plus` teacher workflow.

The formal MGIT release is
`data/releases/cogtrack_vlt_v640_state_update_mgit_segments_v1`: 734 labels across 91 sequences,
350 updates and 384 present hard-nulls, split 645/89, with zero `masked_unknown`. The completed
combined release is `data/releases/cogtrack_vlt_v640_state_update_sft_combined_3063_v1`: 3,063
rows across 315 sequences, split 2,861/202, with 2,253 updates, 810 verified hard-nulls, and zero
`masked_unknown`.

After both primary releases are complete, an optional selective annotation pass may label clearly
identified disappearance/reappearance cases already present in `tracking_sft`. Store those labels as
a sample-id overlay or derived release; never mutate the original tracking release, never duplicate
masked and labelled versions in one training mix, and do not couple this pass to the independent
`state_update_sft` generation.

## Canonical entrypoints

```text
scripts/generate_tracking_sft_data.sh
scripts/generate_state_update_sft_data.sh
scripts/generate_state_update_api_bundle.sh
scripts/modelscope_state_update_transfer.sh
scripts/build_state_update_sft_release.sh
scripts/build_mixed_sft_release.sh
scripts/train_qwen3vl_4b_tracking_sft.sh
```

The only general v6 renderer is `tracking/synthesize_vlt_v6_dataset.py`. Do not reintroduce old
prototype generation entrypoints.

The current additional-label path packages selected three-image cases and uses
`tracking/annotate_state_update_openai_api.py` with an OpenAI-compatible endpoint. Each
present case receives one strong teacher call; `uncertain`, low-confidence, malformed, physically
identity-drifting, and vacuous responses are dropped. The Image-1 visual box is the permanent
identity anchor; the supplied initial text is immutable provenance but may be coarse or wrong, so a
later memory update may substantially correct it without being identity drift. Dataset GT
deterministically labels disappearance transitions; a present case following an absent memory must
generate a reappearance referring expression. Prompt 2.2.1 labels each image role in the API
message, requires evidence to start from Image 3, and suppresses updates for incidental direction
or background changes. Reports use
`annotation_policy=single_pass_frontier_api_v1`, `quality_gate_applied=true`, `dry_run=false`, and
`minimum_output_reached=true`. Do not describe this policy as independent verification.

## Historical release

The completed v6.3.1 release remains available only as a historical baseline:

```text
data/releases/cogtrack_vlt_v631_core_r80_20_case20_robust15_v1
```

It contains 50,220 unique cases and 57,426 views, but uses old Prompt/output keys. Never overwrite,
rename, or cross-load it as v6.4 data.

This server's MGIT mirror has 105 official tiny/train names, 10 missing frame directories and 4
empty directories; exactly 91 sequences are usable. Do not claim 95 or 105 usable sequences.

## Immediate work

1. Finish the running 2×H100 frozen-ViT full SFT and record its validation/checkpoint report.
2. Load the resulting full checkpoint through both local HF and vLLM Prompt-6.4 configurations.
3. Finish the Prompt-6.4 online inference alignment and its no-GT-leakage smoke.
4. Compare Base, old Stage-2, and the new model on the same CognitiveBench-Tiny v6.4 setup.
5. Keep the optional memory overlay deferred until the SFT comparison justifies its cost.
6. TU-GRPO comes only after SFT and state-update evaluation.

Do not claim accuracy improvement from data generation, processor replay, or finite loss. Only a
frozen CognitiveBench comparison establishes tracking improvement.

## Leakage and development constraints

- Training uses Qwen3-VL relative norm1000 boxes and official ms-swift `<bbox> + objects.bbox +
  image_id` with `QWENVL_BBOX_FORMAT=new`.
- Qwen2.5 absolute-pixel views never cross-load into Qwen3 training.
- Reference/history frames are same-sequence present frames strictly earlier than current.
- Negatives are real same-sequence absent frames. Current is never boxed.
- Online inference uses GT only at initialization; later current/future GT never enters the tracker.
- Split by complete sequence.
- Current docs live at top-level `docs/`; historical designs live in `docs/archive/`.
- Prompt text lives only in `cogtrack/prompts/`; paths live in config/environment variables.
- Data, models, outputs, checkpoints, and caches do not enter Git.
- Preserve user changes; never use destructive reset/checkout to clean a dirty worktree.
- Formal state-label reports must declare `minimum_output_reached=true` (default minimum 1,200).
- Before committing, run Ruff, complete active tests, and relevant processor/training preflight.
