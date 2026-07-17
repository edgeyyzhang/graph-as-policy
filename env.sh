# Local dev environment for gap (Vertex-backed LLM + driver/CUDA workaround).
# This must be SOURCED, not executed: `source env.sh` (or `. env.sh`).
# Running it with `./env.sh` or `bash env.sh` would export into a throwaway
# subshell and have no effect on your interactive shell.

# --- gap's LLM client: route through Vertex AI ---

export GAP_LLM_PROVIDER=vertex
export GOOGLE_CLOUD_PROJECT=autolab-res
export ANTHROPIC_VERTEX_PROJECT_ID=autolab-res
export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=global
# vertex is ambiguous between Claude and Gemini models; both `gap generate`
# and the `vlm` tool bundle (which falls back to GAP_LLM_MODEL when
# GAP_VLM_MODEL is unset) need this set explicitly.
export GAP_LLM_MODEL=claude-opus-4-8
export GAP_VLM_MODEL=gemini-2.5-flash
export GAP_LLM_PROVIDER=vertex
export GOOGLE_CLOUD_PROJECT=autolab-res # vertex needs this + gcloud ADC
export MUJOCO_GL=egl                  # sim runs
export CUDA_HOME=/usr/local/cuda      # curobo
# export HF_TOKEN=<token>               # gated weights
export UV_NO_SYNC=1
source ~/.bashrc