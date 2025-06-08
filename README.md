# RAREval-data-processing
This repository provides the official code for the data preprocessing procedures introduced in our paper:
**Do Reviews Matter for Recommendations in the Era of Large Language Models?**


RAREval is a standardized evaluation framework designed to assess the robustness and sensitivity of review-aware recommender systems under varying textual conditions, including:
- **No-Review**: Removes all textual reviews  
- **Reduction**: Randomly removes a portion of reviewText (0%，25%, 50%, 75%，100%)  
- **Distortion**: Reassigns reviewText among user-item pairs (0%，25%, 50%, 75%, 100%)  
- **Data Sparsity**: Applies k-core filtering and splits into train/valid/test
- **Cold-Start**: Evaluates performance across user groups with 1–10 historical interactions (no preprocessing required)
## Requirements

Tested on Python 3.9

1. [pandas](https://pypi.org/project/pandas/) >= 1.5  
2. [numpy](https://pypi.org/project/numpy/) >= 1.23

## Dataset

You need to prepare the following datasets before running RAREval:

1. **Amazon Review Data**  
   Download from the official Amazon data page:  
   🔗 [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)  

   Please download the following **eight domains**, including:
   - The **5-core version**
   - The **raw/original version** (no k-core filtering)

   | Dataset              | Example Filename                          |
   |----------------------|--------------------------------------------|
   | Musical Instruments  | `reviews_Musical_Instruments_5.json`      |
   | Amazon Instant Video |   `reviews_Amazon_Instant_Video_5.json`     |
   | Digital Music        | `reviews_Digital_Music_5.json`             |
   | Video Games          | `reviews_Video_Games_5.json`              |
   | Office Products      | `reviews_Office_Products_5.json`          |
   | Health and Personal Care |`reviews_Health_and_Personal_Care_5.json` |
   | CDs and Vinyl        | `reviews_CDs_and_Vinyl_5.json`            |
   | Movies and TV        | `reviews_Movies_and_TV_5.json`            |
