DSE_511_Final_Project
==============================

Using Post Content to Determine Personality Type

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── raw data 	             <- The original, immutable data dump.
    ├── main.py                  <- model testing, obtain all experiment results 
    ├── hyp_tune_ALL.py            	           <- hyper parameter tuning for n=16 problem   
    ├── hyp_tune_lSVM_4x.py                  <- hyper parameter tuning for n=2 Linear SVM 
    ├── hyp_tune_log_reg_4x.py               <- hyper parameter tuning for n=2 logistic regression
    ├── hyp_tune_rf_4x.py                         <- hyper parameter tuning for n=2 random forest
    ├── hyp_tune_xbg_4x.py                     <- hyper parameter tuning for n=2 XGBBoost
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
    │   │   └── clean_text.py            <- cleans raw posts
    │   │   └── encode.py                 <-encodes MBTI labels to numbers
    │   │   └── dimension_4x.py       <- separate text out to create 4 data sets for each mbti dimension
    │   │   └── train_val_test.py       <- split out dataset for train, validation and testing 
    │   │   
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── __init__.py    
    │   │   └── extraction.py              <- feature extraction using TfidfVectorizer
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions                
    │   │   └── rf_hyp.py                    <-callable function to tune random forest
    │   │   └── XGB_hyp.py               <-callable function to tune XGBoost
    │   │   └── log_reg_hyp.py          <-callable function to tune Logistic Regression 
    │   │   └── lSVC_hyp.py              <-callable function to tune LinearSVM
    │   │   └── model_eval.py            <-callable function to evaluate test data in main 
    │   │   ├── results           <- pickled results from test data set of all models for both n=16 and n=2 problems      
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
