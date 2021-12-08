import pandas as pd
import numpy as np
import re

def clean_mbti(data):
    label = data['type']

    #lowercase
    data['posts'] = data['posts'].apply(lambda x: x.lower()) 
    
    personalities_list = ['enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp',
                          'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp']
    #remove links
    data['posts'] = data['posts'].apply(lambda x: re.sub(r'https?:\/\/.*?[\s+]', '', x.replace("|"," ") + " ")) #links
    
    #remove MBTI personality labels from data['posts']
    for i in range(len(personalities_list)-1):
        data['posts'] = data['posts'].str.replace(personalities_list[i], '')
    
    #remove nonwords
    data['posts'] = data['posts'].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',x))
    
    #remove puncuation
    data['posts'] = data['posts'].apply(lambda x: re.sub(r'[\.+]', ".",x))
    
    #remove extra spaces
    data['posts'] = data['posts'].str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ') 
    
    clean = data
    
    return clean