import sys, os
sys.path.append(os.pardir)
import numpy as np
import utils.ymr_data as ymr
from sklearn import metrics

# Load data
df = ymr.load_ymr_data()
vocab, vocab_inv = ymr.make_vocab(df)
data = ymr.make_polar(df)
train, test = ymr.train_test_split(data)

# Train classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

nb = Pipeline([
        ('vect', CountVectorizer(analyzer='char', ngram_range=(1,3))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())])

nb.fit(train.text, train.rating)
predicted = nb.predict(test.text)
print(metrics.classification_report(test.rating, predicted))

