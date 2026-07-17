# LoRA Fine-Tuning Pipeline

> Fine-tune Llama-3-8B on consumer GPUs in 3 hours.

QLoRA 4-bit + gradient checkpointing for 8B models on 24GB VRAM, targets specific projection layers, W&B sweep for hyperparameter search, one-command deploy to vLLM.

**Part of [AEGIS](https://github.com/KnigguKniggu-droid/AEGIS)** — Adaptive AI Governance Infrastructure for Cyber-Physical Systems. This subsystem maps to **L1: Compute Fabric** (SVD-based model compression — Low-Rank Adaptation approximates weight updates via factored matrices, reducing parameter count by 10-100x.).

---

## Architecture Position

```
AEGIS Layer: L1: Compute Fabric
ECE Mapping: SVD-based model compression — Low-Rank Adaptation approximates weight updates via factored matrices, reducing parameter count by 10-100x.
```

This module is one of 15 subsystems in the AEGIS platform. See the [unified architecture](https://github.com/KnigguKniggu-droid/AEGIS#readme) for how all components interconnect.

---

## Features

- QLoRA 4-bit + gradient checkpointing: 8B model on 24GB VRAM
- Targets q_proj/v_proj/k_proj/o_proj + gate/up/down; rank 64, alpha 16
- WandB sweep for LR/rank/alpha; early stopping on eval loss
- Catastrophic forgetting eval on 5 holdout tasks pre/post
- One-command deploy: modal run serve.py to vLLM endpoint

---

## Tech Stack

`Python` | `PEFT` | `TRL` | `bitsandbytes` | `Weights & Biases` | `vLLM` | `Modal` | `HuggingFace`

---

## Quick Start

```bash
git clone https://github.com/KnigguKniggu-droid/10-lora-finetuning-pipeline.git
cd 10-lora-finetuning-pipeline
pip install -e .
```

Run tests:

```bash
pytest tests/ -v
```

---

## Project Structure

```
10_lora_finetuning_pipeline/
  src/                  # Core application code
  tests/                # 29 passing tests
  .github/              # CI/CD workflows
  Dockerfile            # Container build
  pyproject.toml        # Package configuration
```

---

## Running in Docker

```bash
docker build -t 10_lora_finetuning_pipeline .
docker run -p 8000:8000 10_lora_finetuning_pipeline
```

---

## ECE Design Principles

This subsystem is modeled after classical electrical and computer engineering concepts:

> **SVD-based model compression — Low-Rank Adaptation approximates weight updates via factored matrices, reducing parameter count by 10-100x.**

The AEGIS platform applies safety-critical engineering principles from integrated circuit design to LLM deployment, ensuring production reliability in autonomous vehicles, power grids, and medical devices.

---

## Related Projects

All 15 AEGIS subsystems:

| # | Project | Layer | ECE Mapping |
|---|---------|-------|-------------|
| 01 | [Model Regression Detection](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | SPC |
| 02 | [LLM Cost Autopilot](https://github.com/KnigguKniggu-droid/AEGIS) | L1 | DVFS |
| 03 | [Failure Forensics](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | BIST+ATPG |
| 04 | [Self-Healing Docs](https://github.com/KnigguKniggu-droid/AEGIS) | L6 | ECO |
| 05 | [Output Arbitration](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | TMR |
| 06 | [Hybrid Search RAG](https://github.com/KnigguKniggu-droid/AEGIS) | L3 | Sensor Fusion |
| 07 | [Semantic Cache](https://github.com/KnigguKniggu-droid/AEGIS) | L2 | CAM |
| 08 | [SQL Guardrails](https://github.com/KnigguKniggu-droid/AEGIS) | L4 | MPU/MMU |
| 09 | [A/B Testing](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | SPRT |
| 10 | [LoRA Pipeline](https://github.com/KnigguKniggu-droid/AEGIS) | L1 | SVD |
| 11 | [API Gateway](https://github.com/KnigguKniggu-droid/AEGIS) | L2 | Token Bucket |
| 12 | [Feature Flags](https://github.com/KnigguKniggu-droid/AEGIS) | L5 | FPGA Reconfig |
| 13 | [Dataset Generator](https://github.com/KnigguKniggu-droid/AEGIS) | L3 | Signal Conditioning |
| 14 | [Workflow Orchestrator](https://github.com/KnigguKniggu-droid/AEGIS) | L6 | FSM Sequencer |
| 15 | [LLM Observability](https://github.com/KnigguKniggu-droid/AEGIS) | L7 | Oscilloscope+SA |

---

## License

MIT
