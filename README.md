# RAREval-data-processing
![python](https://img.shields.io/badge/python-%3E=3.8-blue)
![pandas](https://img.shields.io/badge/pandas-%3E=1.5-blue)
![numpy](https://img.shields.io/badge/numpy-%3E=1.23-blue)
![transformers](https://img.shields.io/badge/transformers-4.45.2-orange)
![peft](https://img.shields.io/badge/peft-0.12.0-green)
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


## 📂 Dataset

You need to prepare the following datasets before running RAREval.

🔗 Download from: [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  

Please download both the **5-core** and **raw** versions (no k-core filtering) for the following 8 domains:

| Dataset                  | Example Filename                          |
|--------------------------|--------------------------------------------|
| Musical Instruments      | `reviews_Musical_Instruments_5.json`      |
| Amazon Instant Video     | `reviews_Amazon_Instant_Video_5.json`     |
| Digital Music            | `reviews_Digital_Music_5.json`            |
| Video Games              | `reviews_Video_Games_5.json`              |
| Office Products          | `reviews_Office_Products_5.json`          |
| Health and Personal Care | `reviews_Health_and_Personal_Care_5.json` |
| CDs and Vinyl            | `reviews_CDs_and_Vinyl_5.json`            |
| Movies and TV            | `reviews_Movies_and_TV_5.json`            |

---

## 🔬 Experiment Examples

To run each RAREval experiment, simply execute the corresponding script:

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
### 📌 Cold-Start

This setting requires **no preprocessing script**.  
Cold-start evaluation is performed directly during model evaluation, by grouping test users according to their training interaction count (e.g., 1–10).

An example output for the *Musical Instruments* domain is shown below:
```text
# Output Summary (Musical Instruments)
Training Frequency 0:  MSE = nan,     MAE = nan,     Count = 0
Training Frequency 1:  MSE = 0.5833,  MAE = 0.4167,  Count = 12
...
Training Frequency 10: MSE = 1.7273, MAE = 0.7273,Count = 22
```
📄 Full output available in [examples/cold_start_example.txt](examples/cold_start_example.txt)





