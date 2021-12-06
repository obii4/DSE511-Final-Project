DSE_511_Final_Project
==============================

Using Post Content to Determine Personality Type

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── raw data 	      <- The original, immutable data dump.
    ├── main.py            <- model evaluation 
    ├── hyp_tune_ALL_log_reg.py            <- hyper parameter tuning for n=16 Logistic Regression    
    ├── hyp_tune_ALL_lSVM.py                <- hyper parameter tuning for n=16 Linear SVM 
    ├── hyp_tune_EI_log_reg.py               <- hyper parameter tuning for E/I Logistic Regression 
    ├── hyp_tune_EI_SVM.py                   <- hyper parameter tuning for E/I Linear SVM 
    ├── hyp_tune_NS_log_reg.py             <- hyper parameter tuning for N/S Logistic Regression
    ├── hyp_tune_NS_SVM.py                 <- hyper parameter tuning for N/S Linear SVM 
    ├── hyp_tune_TF_log_reg.py            <- hyper parameter tuning for T/F Logistic Regression
    ├── hyp_tune_TF_SVM.py                <- hyper parameter tuning for T/F Linear SVM 
    ├── hyp_tune_JP_log_reg.py            <- hyper parameter tuning for J/P Logistic Regression  
    ├── hyp_tune_JP_SVM.py                <- hyper parameter tuning for J/P Linear SVM   
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
