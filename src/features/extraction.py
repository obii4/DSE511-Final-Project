from sklearn.feature_extraction.text import TfidfVectorizer
# process raw text into ML compatible features

def feature_Tfidf(X):
    vectorizer = TfidfVectorizer(min_df=3, stop_words='english',
                                 ngram_range=(1, 2), lowercase=True)
    vectorizer.fit(X)

    vec = vectorizer.transform(X)

    return vec