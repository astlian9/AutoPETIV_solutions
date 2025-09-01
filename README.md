# AutoPET-IV-Solution

This repository contains our solution for the [MICCAI25 AUTOPET-IV challenge](https://autopet-iv.grand-challenge.org).

---

## Installation

**Requirements:**  
- Python ≥ 3.10.12 and < 3.12

**Setup steps:**

1. Create a Python (or conda) virtual environment:

    ```bash
    python -m venv autopet
    source autopet/bin/activate
    ```

2. Clone the repository:

    ```bash
    gh repo clone astlian9/AutoPETIV_solutions
    cd AutoPETIV_solutions
    ```

3. Install dependencies:

    ```bash
   cd nnunet-baseline
    pip install -r requirements.txt
    ```

---

## Dataset Download

To replicate or expand upon our experiments, download the AMOS-MM dataset from [here](https://era-ai-biomed.github.io/amos/dataset.html#download). Once downloaded, you can proceed with dataset preparation.

---

## Data Preparation

PET and CT images

Human clicks
The dataset requires converting the JSON file to a heatmap saved as .nii.gz file. To generate it, run the following command:

```bash
python prepare_data.py \
  --report_json <PATH_TO_report_generation_train_val.json> \
  --vqa_json <PATH_TO_vqa_train_val.json> \
  --output <PATH_TO_OUTPUT_DIR> \
  --train_src <PATH_TO_imagesTr> \
  --val_src <PATH_TO_imagesVa>
```

---

## Training

### Medical Report Generation (MRG)

Once data preparation is complete, train the LLaMA 3.1 model for report generation using:

```bash
PYTHONPATH=. accelerate launch --num_processes 1 --main_process_port 29500 LaMed/src/train/amos_train.py \
    --version v0 \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --cache_dir <WHERE_MODEL_WILL_BE_SAVED> \
    --model_type llama \
    --freeze_llm True \
    --vision_tower vit3d \
    --pretrain_vision_model <PATH_TO_PRETRAINED_VISION_MODEL> \
    --bf16 True \
    --output_dir <WHERE_TO_SAVE_MODEL> \
    --num_train_epochs 100 \
    --per_device_train_batch_size 2 \
    --evaluation_strategy "no" \
    --do_eval False \
    --eval_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True \
    --dataloader_num_workers 4 \
    --report_to none \
    --prompt "simple" \
    --task mrg \
    --json_path <PATH_TO_DATASET_JSON> \
    --image_size "32, 256, 256" \
    --with_template True \
    --model_max_length 768
```

- The `json_path` should point to the JSON file prepared earlier.
- Set `cache_dir` and `pretrain_vision_model` appropriately.
-  Additional arguments:
  - `zoom_in`: uses organ segmentation masks for region cropping.
  - `prompt`: controls the prompt format (e.g. `"simple"` in `LaMed/src/dataset/prompts.py`).

---
## Model weight
Model weight is release by using [Google Drive](https://drive.google.com/drive/folders/1kqSx4cYmDMgVUs2DQMDiQcw9j8rTXSxJ?usp=share_link)
## Inference

### MRG Inference

Run inference for medical report generation:

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch --num_processes 1 --main_process_port 29500 infer.py \
  --model_name_or_path <PATH_TO_TRAINED_MODEL> \
  --json_path <PATH_TO_DATA_JSON> \
  --model_max_length 768 \
  --prompt "simple" \
  --post_process "normality" "bq" \
  --triplet_model_path <PATH_TO_TRAINED_TRIPLET_MODEL> \
  --proj_out_num 256
```

**Note:**  
- If you did not train a triplet model, omit the `"bq"` argument and `--triplet_model_path`.
- The `post_process` argument enables:
  - Knowledge-based normality inference.
  - Focused questioning based on specific findings.
- The knowledge base is defined in `utils/postprocessor.py`. Adapt it for different datasets.

---

## Wrap up to Docker



## Acknowledgements

- We thank the organizers of the MICCAI25 AutoPET challenge for their efforts.
- This codebase builds upon the [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/tree/master), and we gratefully acknowledge its authors.

