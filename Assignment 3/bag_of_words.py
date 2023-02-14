from sklearn.feature_extraction.text import CountVectorizer

bruh= [
'This is the first document document document.',    'This document is the second document.',     'And this is the third one.',
    'Is this the first document?',
 ]

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(bruh)
print(vectorizer.get_feature_names())
print(X.toarray()[0])




