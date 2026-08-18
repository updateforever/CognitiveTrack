# Historical local state-teacher workflow

This directory preserves the superseded local Qwen3-VL-32B teacher plus independent-verifier
workflow and the earlier MGIT label CLI for reproducibility. They are not current CognitiveTrack
entrypoints.

The v6.4 production path uses:

- `tracking/plan_mgit_state_update_data.py` for reliable MGIT action-segment labels;
- `tracking/annotate_state_update_openai_api.py` for the portable frontier API workflow;
- `scripts/generate_state_update_api_bundle.sh` and `scripts/build_state_update_sft_release.sh` as
  the maintained shell entrypoints.

Do not use the scripts in this directory to regenerate the current formal releases. Historical
commands remain runnable only to audit older teacher/verifier artifacts.
