import pandas as pd
import numpy as np
import re

def text_split(text):
    extroversion = text[text['type'].isin(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'])]
    extroversion = extroversion.replace(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])
    introversion = text[text['type'].isin(['INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'])]
    introversion = introversion.replace(['INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])

    intuition = text[text['type'].isin(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'INTJ', 'INTP'])]
    intuition = intuition.replace(['ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'INTJ', 'INTP'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])
    sensing = text[text['type'].isin(['ISFJ', 'ISFP', 'ISTJ', 'ISTP','ESFJ', 'ESFP', 'ESTJ', 'ESTP'])]
    sensing = sensing.replace(['ISFJ', 'ISFP', 'ISTJ', 'ISTP','ESFJ', 'ESFP', 'ESTJ', 'ESTP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])

    thinking = text[text['type'].isin(['ENTJ', 'ENTP','ESTJ', 'ESTP','INTJ', 'INTP', 'ISTJ', 'ISTP'])]
    thinking = thinking.replace(['ENTJ', 'ENTP','ESTJ', 'ESTP','INTJ', 'INTP', 'ISTJ', 'ISTP'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])
    feeling = text[text['type'].isin(['ENFJ', 'ENFP','ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'])]
    feeling = feeling.replace(['ENFJ', 'ENFP','ESFJ', 'ESFP', 'INFJ', 'INFP', 'ISFJ', 'ISFP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])

    judging = text[text['type'].isin(['ENFJ','ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'])]
    judging = judging.replace(['ENFJ','ENTJ', 'ESFJ', 'ESTJ', 'INFJ', 'INTJ', 'ISFJ', 'ISTJ'], 
                                    [0, 0, 0, 0, 0, 0, 0, 0])

    percieving = text[text['type'].isin(['ENFP', 'ENTP', 'ESFP', 'ESTP', 'INFP', 'INTP', 'ISFP', 'ISTP'])]
    percieving = percieving.replace(['ENFP', 'ENTP', 'ESFP', 'ESTP', 'INFP', 'INTP', 'ISFP', 'ISTP'], 
                                    [1, 1, 1, 1, 1, 1, 1, 1])
    
    EI = pd.concat([extroversion, introversion])
    NS = pd.concat([intuition, sensing])
    TF = pd.concat([thinking, feeling])
    JP = pd.concat([judging, percieving])
    
    return EI, NS, TF, JP