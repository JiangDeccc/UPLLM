# UPLLM: Collaborative-Signal Enhanced LLM-Based User Profiling

This repository contains the implementation of **UPLLM**, a collaborative-signal enhanced framework for LLM-based user profiling and recommendation.

UPLLM combines three components:

- LLM-based iterative user profile generation and refinement.
- Collaborative signals extracted from recommendation models.
- A customized RecBole backbone for training, evaluation, and profile-aware recommendation experiments.

## Repository Structure

```text
UPLLM/
├── Datasets/                         # Processed example datasets kept with the repository
│   ├── amazon-CDs_and_Vinyl/
│   └── ml-25m/
├── recbole/                          # Customized RecBole backbone
├── agent_vllm.py                     # Main LLM profiling implementation
├── potential_interest.py             # Potential-interest mining and profile update pipeline
├── collaborative_infos.py            # Collaborative-signal utilities and embedding/profile helpers
├── rec.py                            # Recommendation training, tuning, and evaluation entry point
├── best_params.py                    # Hyperparameter grids and selected settings
├── baseline.py                       # Baseline embedding/profile generation helpers
├── create_new_dataset.py             # Dataset conversion and RecBole-format generation helpers
├── dataprocessing.py                 # Raw data preprocessing utilities
├── embedmulti.py                     # Embedding-based metric utilities
├── load.py                           # Lightweight data loading helpers
├── prompt.py                         # Prompt templates for profile generation and refinement
├── sensitivity.py                    # Sensitivity, ablation, and log-analysis utilities
├── mylogging.py                      # Local logging helper
├── requirements.txt                  # Python dependency snapshot
└── README.md
```

## Main Components

### `agent_vllm.py`

Implements the LLM-based user profiling workflow, including:

- user and item profile objects;
- iterative profile update and condensing;
- profile rollback/checkpoint utilities;
- embedding generation and profile similarity helpers;
- LLM evaluation utilities.

The default LLM client is OpenAI-compatible and points to a local vLLM-style service. Update `api_key`, `base_url`, `MODEL_NAME`, and `EMBED_MODEL_NAME` according to your runtime environment before running full profiling jobs.

### `potential_interest.py`

Builds potential-interest signals from recommendation outputs and injects them back into user profiles. It includes utilities for:

- selecting potential items;
- summarizing potential interests with an LLM;
- modifying user profiles;
- generating profile embeddings;
- retrying failed users and appending missing embeddings.

### `rec.py`

Provides the main recommendation-side experiment workflow on top of the customized RecBole code. It supports dataset preparation, embedding preparation, training/evaluation, parameter tuning, and grouped evaluation.

Example command:

```bash
python rec.py \
  --dataset_type ml-25m \
  --dataset_choice user_5k_total_ml-25m_5_32_full \
  --model XSimGCL \
  --gpu_id 0 \
  --tuning_param Complement
```

For Amazon CDs and Vinyl experiments, use `--dataset_type amazon-CDs_and_Vinyl` and the matching RecBole dataset folder name as `--dataset_choice`.

### `prompt.py`

Stores the prompt templates used for user-profile update, condensing, potential-interest summarization, and profile modification.

## Data Layout

The repository keeps processed example data under `Datasets/`. Some experiment scripts also expect working directories such as:

```text
./ml-25m/user_5k/
./amazon/user_5k/CDs_and_Vinyl/
./Embeddings/
./logs/
./sortdict/
```

These directories are runtime artifacts or local working copies. They are intentionally ignored by Git when they contain generated embeddings, logs, checkpoints, or temporary files.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The dependency file is a snapshot of the experiment environment. GPU/CUDA-specific packages may need adjustment for your local CUDA and PyTorch versions.

## Notes on LLM Services

The LLM-related scripts use OpenAI-compatible clients. For local vLLM servers, update the client settings in `agent_vllm.py` and `potential_interest.py`, for example:

```python
base_url="http://localhost:8003/v1"
```

Embedding calls use a separate OpenAI-compatible client by default.

## RecBole Acknowledgement

This project builds on the open-source RecBole framework:

> RecBole: A Unified Recommendation Framework  
> https://github.com/RUCAIBox/RecBole

The `recbole/` directory keeps the upstream structure while adding project-specific changes for collaborative-signal extraction, profile-aware recommendation, and customized evaluation.

