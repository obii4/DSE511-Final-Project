DSE_511_Final_Project
==============================

Using Post Content to Determine Personality Type

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── raw data 	      <- The original, immutable data dump.
    ├── main.py            <- model evaluation 
    ├── hyp_tune_ALL.py            	         <- hyper parameter tuning for n=16 problem   
    ├── hyp_tune_lSVM_4x.py                  <- hyper parameter tuning for n=2 Linear SVM 
    ├── hyp_tune_log_reg_4x.py               <- hyper parameter tuning for n=2 logistic regression
    │
    ├── notebooks          <- Jupyter notebooks that contain initial data data exploration
    │   └── o'brien exploration.ipynb    
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         		       generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── __init__.py
    │   │   └── load.py
    │   │   └── clean_text.py 
    │   │   └── encode.py <-encodes MBTI labels to numbers
    │   │   └── dimension_4x.py <- separate text out to create 4 data sets for each mbti dimension
    │   │   
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── __init__.py    
    │   │   └── extraction.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
