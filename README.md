DSE_511_Final_Project
==============================

Using Post Content to Determine Personality Type

To run code, download the data and place it on your desktop.

Project Organization
------------

    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── raw data 	           <- The original, immutable data dump.
    ├── main_n16.py            <- model testing for n=16 problem, obtain all experiment results 
    ├── main_n2.py             <- model testing for n=2 problem, obtain all experiment results 
    ├── roc_n16.py                   <- generate roc curves for n=16 problem
    ├── roc_n2.py                    <- generate roc curves forn=2 problem
    ├── summarize_results.py         <- generate tables that show accuracy, train/test times
    ├── misclassifications.py        <- misclassified terms for each model for n=16 problem
    ├── hyp_tune_ALL.py            	 <- hyper parameter tuning for n=16 problem   
    ├── hyp_tune_lSVM_4x.py          <- hyper parameter tuning for n=2 Linear SVM 
    ├── hyp_tune_log_reg_4x.py       <- hyper parameter tuning for n=2 logistic regression
    ├── hyp_tune_rf_4x.py            <- hyper parameter tuning for n=2 random forest
    ├── hyp_tune_xbg_4x.py           <- hyper parameter tuning for n=2 XGBBoost
    │
    ├── notebooks          <- Jupyter notebooks that contain initial data data exploration
    │   └── o'brien exploration.ipynb    
    │
    ├── reports            
    │   ├── O'Brien, Sanford, Pike DSE 511 Project Proposal .pdf    <- project proposal
    │   ├── dse511_final_report.pdf    <- final report write up
    │   ├── dse511_final_prez.pdf      <- final presentation slides   
    │
    ├── requirements.txt   <- python version == 3.8.8
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
    │   │   └── rf_hyp.py                <-callable function to tune random forest
    │   │   └── XGB_hyp.py               <-callable function to tune XGBoost
    │   │   └── log_reg_hyp.py           <-callable function to tune Logistic Regression 
    │   │   └── lSVC_hyp.py              <-callable function to tune LinearSVM
    │   │   └── model_eval.py            <-callable function to evaluate test data in main 
    │   │   ├── results    <- pickled results from test data set of all models for both n=16 and n=2 problems     
    │   │   		└── exp_problem_times.pkl            <- contains train and test times 
    │   │   		└── exp_problem_class_results.pkl    <- contains classification report 
    │   │   		└── exp_problem_labels.pkl           <- contains ground truth, prediction, prediction probability   
    │   │
    │   └── visualization  
    │   │   ├── tables                  <- accuracy, train/test times for both n=16 and n=2 problems     
    │   │   ├── roc                     <- roc curves for all models and problems
    │   │   ├── exploration             <- data visualization 
    │       
    │
