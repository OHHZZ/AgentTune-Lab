# Project Plan

This repository turns the course labs into a small local fine-tuning project.

## Mapping from labs to project files

- Data preparation labs -> `scripts/prepare_data.py`
- Training lab -> `scripts/train_lora.py`
- Evaluation lab -> `scripts/evaluate.py`
- Inference examples -> `scripts/infer.py`
- Original exported notes -> `docs/course_labs/`

## Scope

The project intentionally uses a tiny default model, `EleutherAI/pythia-70m-deduped`, so the full pipeline can be tested on modest hardware. For a stronger resume demo, replace the model and data with a domain-specific dataset, for example Qwen plus Chinese resume/JD matching examples.
