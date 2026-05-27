# AGENTS.md — KV Packet

Guidance for AI agents working in this repository.

## Critical: use `uv` for everything Python

**Always use [uv](https://docs.astral.sh/uv/) to manage dependencies and run scripts.** Do not use bare `python`, `pip install`, or ad-hoc virtualenvs unless the user explicitly asks otherwise.

| Task | Command |
|------|---------|
| Install / sync deps | `uv sync` |
| Add a dependency | `uv add <package>` |
| Run a script | `uv run python <script.py> [args...]` |
| Run a module | `uv run python -m <module> [args...]` |
| Python version | `uv python` (install/list only — not for running project scripts) |

The project is locked via `pyproject.toml` and `uv.lock`. After changing dependencies, run `uv lock` if needed, then `uv sync`.

**Requirements:** Python ≥ 3.12. PyTorch is pinned for CUDA; see `requirements.txt` / comments in `pyproject.toml` if you need the PyTorch wheel index (`cu130`).

---

## What this repo does

**KV Packet** trains small header/trailer adapter vectors around document KV caches so multi-document RAG can concatenate precomputed caches without recomputation. Workflow:

1. **Train** adapters — `run_train_filler.py`
2. **Evaluate** on benchmarks — `run_eval.py`
3. **(Optional)** Build wrapper from handcrafted tokens — `run_build_packet.py`

Core library: `kv_packet/` (`packet_wrapper`, `cache`, `cache_comb`, `dataset`, `model`, `utils`).

Configs:

- Training: `packet_wrapper_config/<model>/<dataset>/...json`
- Evaluation: `eval_config/<model>/<dataset>/...json`

Configs inherit from `_default.json` in the same directory (override files only need differing fields).

Before training or eval, set `model.model_path` in configs to a local path or Hugging Face model id.

---

## Running scripts (examples)

Train adapters (single config):

```bash
uv run python run_train_filler.py packet_wrapper_config/qwen_3_8b/biography/8_8.json
```

Train a directory of configs:

```bash
uv run python run_train_filler.py packet_wrapper_config/llama_3_1_8b/mixture/
```

Evaluate:

```bash
uv run python run_eval.py eval_config/qwen_3_8b/niah/kv_packet.json
uv run python run_eval.py eval_config/qwen_3_8b/biography/ --overwrite --debug
```

Build packet from explicit tokens:

```bash
uv run python run_build_packet.py <config.json>
```

Plotting (under `plot_scripts/`):

```bash
uv run python plot_scripts/draw_main_results.py
```

---

## Agent conventions

- **Scope:** Prefer minimal, focused diffs. Match existing style in `kv_packet/` and entry scripts.
- **Configs:** Do not hardcode model paths; use config JSON. See `res/train_config.md` and `res/eval_config.md` for field reference.
- **Tests:** Only add tests when requested or they cover non-trivial behavior.
- **Docs:** User-facing workflow lives in `README.md`; keep `AGENTS.md` for agent/tooling workflow.
- **Commits:** Only commit when the user explicitly asks.

---

## Key paths

| Path | Purpose |
|------|---------|
| `run_train_filler.py` | Phase 1: train header/trailer |
| `run_eval.py` | Phase 2: benchmark evaluation |
| `run_build_packet.py` | Ablation: init wrapper from tokens |
| `kv_packet/packet_wrapper/` | `PacketWrapper` parameters |
| `kv_packet/cache_comb/` | Cache combination + baselines |
| `packet_wrapper_config/` | Training configs |
| `eval_config/` | Evaluation configs |
| `eval_results/` | Written next to configs as `*_result.json` |

Supported chat templates: `"llama_chat"` (Llama-3.1-8B), `"qwen_3_chat"` (Qwen2.5/3).
