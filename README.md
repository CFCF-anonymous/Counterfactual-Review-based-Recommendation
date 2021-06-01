# Counterfactual-Feature-aware-Collaborative-Filtering

## Requirements

- Python 3.6.9
- PyTorch 1.4.0 + cuda 10 
- numpy 1.16


## Project Structure

    .
    |-- __init__.py
    |-- data               # the directory of dataset:  the preprocessed Amazon Dataset, please follow the "DataStructure" Step.
    |-- main.py            # the training logic of our CFCF method. change the --anchor_model as 1|2|3|4|5 to change to different logic
    |-- config.py          # config the anchor model type, you can set anchor_type to "ele_add | ele_mul | hybrid | attention", and you can set the number of gpus 
    |-- data_loader.py     # dataloader of the Amazon Dataset. 
    |-- model              # the model directory. contain anchor model and the intervener model
    |   |-- __init__.py
    |   |-- anchor_model.py # the f model in our paper, Here is the Multiply anchor model
    |   |-- intervention_model.py # counter factual sample method
    |   |-- anchor_hybrid.py # the f model in our paper, Here is the hybrid anchor model
    |   |-- anchor_attention.py # the f model in our paper, Here is the attention anchor model
    |   `-- anchor_ele_add.py # the f model in our paper, Here is the ele add anchor model
    | 
    `-- utils
        |-- QPC.py    
        `-- eval.py       # inference code, calculate the metrics of recommendation: NDCG F1 HitRate 

## Data Structure
download our dataset and unzip it in ./data/ directory.

here, we provided the small dataset [Amazon_Instant_Video.tar](https://pan.baidu.com/s/1XegBz-Lq3tew_ktvwG_hTA) for test. the extraction code is 9w4m
the complete dataset is 3G [Amazon_dataset_complete](https://pan.baidu.com/s/1t85gLJayObqmuN7DQJCG4A). the extraction code is q4vt

data directory will like following: 

    data
    |-- Amazon_Instant_Video
        |-- Amazon_Instant_Video.formated
        |-- anchor.ptr
        |-- anchor_best.ptr
        |-- feature_id_dict
        |-- id_feature_dict
        |-- id_item_dict
        |-- id_user_dict
        |-- item_feature_quality_matrix
        |-- item_id_dict
        |-- predicted_item_feature_quality
        |-- predicted_user_feature_attention
        |-- sorted_ided_dataset
        |-- statistics
        |-- test_compute_user_items_dict
        |-- test_data
        |-- test_ground_truth_user_items_dict
        |-- train_data
        |-- train_user_negative_items_dict
        |-- train_user_positive_items_dict
        |-- user_feature_attention_matrix
        `-- user_id_dict

## Usage

1. Install all the required packages

2. Unzip the dataset in ./data directory and check every file is exists.

4. modify the config.py, set the anchor model to what you want. default is "ele_mul" anchor model : CF-mul

4. Run 
```
python main.py --data_path=./data/Amazon_Instant_Video/ --anchor_model=1  # train the anchor model and save it
mv ./data/Amazon_Instant_Video/anchor.ptr ./data/Amazon_Instant_Video/anchor_best.ptr  # change name of saved model to 'anchor_best.ptr'
python main.py --data_path=./data/Amazon_Instant_Video/ --anchor_model=2 --confidence=0.55 --intervener_learning_rate=0.001 --intervener_reg=0.01 --learning_rate=0.001 --intervener_feature_number=60 --intervener_l1_reg=0.0025          # generate the counterfactual sample and finetune the anchor model(CF-Base)
```

## Result

CF-Base for AmazonInstantVideo :  0.088
CF-Hard for AmazonInstantVideo :  0.097

## Cases

Please See the paper.
