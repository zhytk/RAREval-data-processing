# RAREval-data-processing
This repository provides data preprocessing scripts for our TKDE submission. It supports generating datasets for four evaluation settings under the RAREval framework:

- **No-Review**: Removes all textual reviews  
- **Reduction**: Randomly removes a portion of reviewText (25%, 50%, 75%)  
- **Distortion**: Reassigns reviewText among user-item pairs (25%, 50%, 75%, 100%)  
- **Data Sparsity**: Applies k-core filtering and splits into train/valid/test
