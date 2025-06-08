# RAREval-data-processing
This repository provides the official code for the data preprocessing procedures introduced in our paper:
Do Reviews Matter for Recommendations in the Era of Large Language Models?

RAREval is a standardized evaluation framework designed to assess the robustness and sensitivity of review-aware recommender systems under varying textual conditions, including:
- **No-Review**: Removes all textual reviews  
- **Reduction**: Randomly removes a portion of reviewText (0%，25%, 50%, 75%，100%)  
- **Distortion**: Reassigns reviewText among user-item pairs (0%，25%, 50%, 75%, 100%)  
- **Data Sparsity**: Applies k-core filtering and splits into train/valid/test
