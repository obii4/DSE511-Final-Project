import pandas as pd
import numpy as np
import re

def clean_mbti_text(data):
    label = data['type']
    
    personalities = np.unique(np.array(data['type']))
    personalities_list = personalities.tolist()
    personalities_list = [p.lower() for p in personalities_list]
    
    #remove links
    data['posts'] = data['posts'].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|"," ") + " ")) #links
    
    #remove MBTI personality labels from data['posts']
    for i in range(len(personalities_list)-1):
        data['posts'] = data['posts'].str.replace(personalities_list[i], '')
    
    #lowercase
    data['posts'] = data['posts'].apply(lambda x: x.lower()) 
    
    #remove nonwords
    data['posts'] = data['posts'].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))
    
    #remove puncuation
    data['posts'] = data['posts'].apply(lambda x: re.sub(r'[\.+]', ".",x)) 
    
    #remove extra spaces
    data['posts'] = data['posts'].str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ') 
    
    clean = data
    
    return clean