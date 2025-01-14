## 1 Environment

Follow [this file](../README.md).

## 2 Dataset

Process FineWeb dataset:

```bash
python mates/process_fineweb.py --base_dir . --split 0
```

- `base_dir` is the place to store the dataset
- `split` is chosen from 0-9

Generate the warmup dataset:

```bash
python mates/select_data.py --method random --ckpt 0 --split 0
```

## 3 Experiments

### 3.1 Pretraining

Our pretraining is run stage by stage to facilitate the model-aware data selection. Each stage consists of 10k steps. For instance, in the initial warmup 10k steps, you can run:

```bash
method=random split=0 ckpt=0 decay=false bash scripts/pretrain.sh
```

### 3.2 Data Selection

After the warmup 10k steps, we can start the MATES data selection process:

1️⃣ Get oracle data influence:

```bash
method=random ckpt=10000 bash scripts/probe_oracle_data_influence.sh
```

- For the warmup checkpoint, `method=random`, but for the following, `method=mates`
- Change the `ckpt` accordingly

2️⃣ Train data influence model:

```bash
ckpt=10000 bash scripts/train_data_influence_model.sh
```

3️⃣ Predict data influence:

```bash
ckpt=10000 bash scripts/predict_data_influence.sh
```

4️⃣ Select the training data for the 10k steps:

```bash
python mates/select_data.py --method mates --ckpt 10000 ---split 0 -warmup true
```

- `warmup` means the data influence is derived from a randomly trained model. We don't need it in the following iterations (where the data influence is derived from a MATES trained model)

5️⃣ Train the MATES model:

```bash
method=mates split=0 ckpt=0 decay=false bash scripts/pretrain.sh
```

- Recommend to change `split` every 10k steps

### 3.3 Evaluation

1️⃣ It is advised to run the evaluation after the decay stage for intermediate checkpoints for better stability.

```bash
method=random split=0 ckpt=10000 decay=true bash scripts/pretrain.sh
```

2️⃣ We provide a simple evaluation example here, and you can modify the parameters based on your needs.

```bash
method=random ckpt=10200 bash scripts/eval.sh
```
