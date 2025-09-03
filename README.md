# AutoPET-IV-Solution

This repository contains our solution for the [MICCAI25 AUTOPET-IV challenge](https://autopet-iv.grand-challenge.org).
Please check Solutions_AutoPETiV.pdf for model details.
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

To replicate or expand upon our experiments, download the AutoPET-IV dataset from [here](https://autopet-iv.grand-challenge.org/dataset/). Once downloaded, you can proceed with dataset preparation.

---

## Data Preparation

### PET and CT images

PET and CT images are processed using nnUNet default preprocessing methods. To generate preprocessed dataset, run

```bash
nnUNetv2_extract_fingerprint -d 140
nnUNetv2_plan_experiment -d 140
nnUNetv2_plan_experiment -d 140 -pl ResEncUnetPlanner
nnUNetv2_preprocess -d 221 -c 3d_fullres
```

### Human clicks
The dataset requires converting the JSON file to a heatmap saved as .nii.gz file. Change the corresponding file path in nnunet-train/process_click.py.  To generate it, run the following command:

```bash
python nnUnet_train/process_click.py
```

---

## Training

### Modify training plan to enable res_encoder

Modify the generated plans.json to allow res_encoder:
```bash
  "3d_fullres_resenc": {
            "inherits_from": "3d_fullres",
            "network_arch_class_name": "ResidualEncoderUNet",
            "n_conv_per_stage_encoder": [
                1,
                3,
                4,
                6,
                6,
                6
            ],
            "n_conv_per_stage_decoder": [
                1,
                1,
                1,
                1,
                1
            ]
        },
"3d_fullres_resenc_bs240": {
            "inherits_from": "3d_fullres_resenc",
            "batch_size": 240
            },
```
### Run training

Once dataset plan is complete, train the nnUNetv2 model using:

```bash
nnUNet_compile=T nnUNet_n_proc_DA=28 nnUNetv2_train 140 3d_fullres_resenc_bs240 0 -num_gpus 8
```

- The `nnUNet_n_proc_DA` is the number of processes of Dataloader, change it if your RAM is not enough.
- The `3d_fullres_resenc_bs240` is the name of training plan, change it if necessary.
- The `n-num_gpus` is the number of gpus.

---
## Model weight
Model weight is release by using [Google Drive.](https://drive.google.com/drive/folders/1kqSx4cYmDMgVUs2DQMDiQcw9j8rTXSxJ?usp=share_link)
Download the weigh and copy it to nnunet-baseline/nnUnet_results
## Inference

### Inference

Run inference for tumor segmentation:

```bash
nnUNetv2_predict -i INPUT -o OUTPUT1 -d 140 -c 3d_fullres_resenc_bs240 -f 0 -step_size 0.6 --save_probabilities
```



## Wrap up to Docker
To submit, the model needs to be transfered to Docker. Run:
```bash
cd nnunet-baseline
bash build.sh
bash export.sh
```


## Acknowledgements

- We thank the organizers of the MICCAI25 AutoPET challenge for their efforts.
- This codebase builds upon the [nnUNet repository](https://github.com/MIC-DKFZ/nnUNet/tree/master), and we gratefully acknowledge its authors.

