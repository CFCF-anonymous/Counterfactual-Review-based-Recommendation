# [Code Submission] Neural Feature-aware Recommendation with Signed Hypergraph Convolutional Network



![image](https://github.com/code4review/TOIS/raw/master/img/intro.png)

## Requirements

- Python 3.7
- PyTorch 1.0
- numpy 1.16


## Project Structure

    .
    ├── conf                                # Config files
    ├── data
        ├── Amazon_Instant_Video            # Experiment data   
    ├── results                             # Results saving
    ├── model          
        ├── SHCN.py                         # Model
        ├── utils.py                        # Some useful functions
    ├── data_loader.py          
    ├── main.py                             # The entrance of the project     

## config.py
1. n\_gpus means the number of gpus in your machine
2. the second means the anchor model type

## Usage

1. Install all the required packages

2. Run python main.py

3. Run Python
