# RAREval-data-processing

## 📘 Overview

This repository provides the official code for the data preprocessing procedures introduced in our paper:  
**Do Reviews Matter for Recommendations in the Era of Large Language Models?**

<p align="center">
  <img src="https://github.com/zhytk/RAREval-data-processing/blob/main/RAREval_framework.png?raw=true" alt="RAREval Framework" width="900"/>
</p>


The figure above presents the RAREval framework, which is designed to evaluate review-aware recommender systems under five distinct experimental conditions:

- **No-Review**: Removes all textual reviews  
- **Reduction**: Randomly removes a portion of review texts (0%, 25%, 50%, 75%, 100%)  
- **Distortion**: Reassigns review texts among user-item pairs (0%, 25%, 50%, 75%, 100%)  
- **Data Sparsity**: Applies k-core filtering and splits into train/valid/test  
- **Cold-Start**: Evaluates performance across user groups with 1–10 historical interactions (no preprocessing required)

This repository provides a suite of scripts that generate modified datasets corresponding to the above five evaluation settings, enabling reproducible benchmarking and robustness analysis of large language model-based recommendation systems.

## 📦 Requirements

Tested on:  
![python](https://img.shields.io/badge/python-%3E=3.8-blue)  

### 📚 Core Libraries  
![pandas](https://img.shields.io/badge/pandas-%3E=1.5-blue)
![numpy](https://img.shields.io/badge/numpy-%3E=1.23-blue)
![scipy](https://img.shields.io/badge/scipy-%3E=1.14.1-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-%3E=1.5.2-blue)
![tqdm](https://img.shields.io/badge/tqdm-%3E=4.66.4-blue)

### 🤗 HuggingFace & LLM Support  
![transformers](https://img.shields.io/badge/transformers-4.45.2-green)
![peft](https://img.shields.io/badge/peft-0.12.0-green)
![accelerate](https://img.shields.io/badge/accelerate-%3E=1.1.1-green)
![huggingface--hub](https://img.shields.io/badge/huggingface--hub-%3E=0.29.3-green)
![datasets](https://img.shields.io/badge/datasets-%3E=3.0.0-green)
![evaluate](https://img.shields.io/badge/evaluate-%3E=0.4.3-green)
![bitsandbytes](https://img.shields.io/badge/bitsandbytes-%3E=0.43.1-green)

### 🧠 Model Runtime Support  
![torch](https://img.shields.io/badge/torch-%3E=2.5.1-red)
![llama_cpp_python](https://img.shields.io/badge/llama__cpp__python-0.2.90-red)
![lm--format--enforcer](https://img.shields.io/badge/lm--format--enforcer-%3E=0.10.10-red)

---

## 📂 Dataset

You need to prepare the following datasets before running RAREval.

🔗 Download from: [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  

Please download the **5-core** version for the following 8 domains into the ```data/``` directory, and likewise, the **raw reviews** or **0-core** version for the same 8 domains into the ```data/raw_reviews``` directory:

| Dataset                  | Example Filename (5-core)         |
|--------------------------|-----------------------------------|
| Musical Instruments      | `Musical_Instruments_5.json`      |
| Amazon Instant Video     | `Amazon_Instant_Video_5.json`     |
| Digital Music            | `Digital_Music_5.json`            |
| Video Games              | `Video_Games_5.json`              |
| Office Products          | `Office_Products_5.json`          |
| Health and Personal Care | `Health_and_Personal_Care_5.json` |
| CDs and Vinyl            | `CDs_and_Vinyl_5.json`            |
| Movies and TV            | `Movies_and_TV_5.json`            |

---

## 🔬 Experiment Examples

To run each RAREval experiment, first execute ```train_valid_test_split.sh``` to split the data into training, validation and test set, as follows

```bash
./train_valid_test_split.sh
```

Then, execute the corresponding scripts:

### 📌 No-Review
```bash
python no_review_processing.py
```

### 📌 Reduction
```bash
python reduction_processing.py
```

### 📌 Distortion
```bash
python distortion_processing.py
```

### 📌 Data Sparsity
```bash
python data_sparsity_kcore.py
```

### 📌 Zero-shot and Few-shot
Example usage (change dataset_dir and llm as appropriate):
```bash
python zero_shot.py --dataset_dir data/true-data/reviews_Musical_Instruments --llm qwen2.5:0.5b
python few_shot.py --dataset_dir data/true-data/reviews_Musical_Instruments --llm qwen2.5:0.5b
python zero_shot_slow.py --dataset_dir data/true-data/reviews_Musical_Instruments --llm qwen2.5:0.5b
python few_shot_slow.py --dataset_dir data/true-data/reviews_Musical_Instruments --llm qwen2.5:0.5b
```

The slow version is only recommended for large language models that are not available in a single GGUF file.

For our experiments, we ran the slow version for large language models containing at least 6 billion parameters, and the fast version otherwise. 

The output will be a .jsonl file on the main project directory showing the user training frequency, the item training frequency, predicted rating, and actual rating. This can be used for cold start analysis, and does not require ```cold_start_evaluation.py```.

### 📌 LLM Summarization


### 📌 LLM LoRA finetuning


### 📌 Cold-Start

This setting requires no preprocessing script.
Cold-start evaluation is performed directly during model evaluation, by grouping test users according to their training interaction count (e.g., 1–10).

We provide a standalone evaluation script to compute MSE, MAE, and sample count for each user group based on their frequency in the training set.

📁 Evaluation script: cold_start_evaluation.py

An example output for the *Musical Instruments* domain is shown below:
```text
# Output Summary (Musical Instruments)
Training Frequency 0:  MSE = nan,     MAE = nan,     Count = 0
Training Frequency 1:  MSE = 0.5833,  MAE = 0.4167,  Count = 12
...
Training Frequency 10: MSE = 1.7273, MAE = 0.7273,Count = 22
```
📄 Full output available in examples/cold_start_example.txt

⚠️ Note: The evaluation script does not perform model inference. It assumes predictions are already generated by your model and saved in JSONL format.





