# -*- coding: utf-8 -*-
# @Time    : 2019/9/21 20:23
# @Author  : uhauha2929
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def load_texts():
    texts = []
    labels = []
    with open("data/tweets/Tweets.csv", "r") as text_file:
        reader = csv.reader(text_file)
        next(reader)
        for line in reader:
            text = line[10].strip().lower()
            if len(text.split()) > 0:  # remove empty line
                texts.append(text)
                label = line[1].strip().lower()
                if label == 'negative':
                    labels.append(0)
                elif label == 'positive':
                    labels.append(1)
                else:
                    labels.append(2)
    return texts, labels


def load_tests():
    texts = []
    labels = []
    line_number = 0
    with open('tweets.txt', 'rt', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            if line_number % 3 == 1:
                labels.append(0)
            elif line_number % 3 == 2:
                labels.append(1)
            else:
                labels.append(2)
            texts.append(line.strip())
    return texts, labels


if __name__ == '__main__':
    train_texts, train_labels = load_texts()
    print(train_texts[0], train_labels[0])

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    print(X_train.shape)

    svm = LinearSVC()
    svm.fit(X_train, train_labels)

    test_texts, test_labels = load_tests()
    X_test = vectorizer.transform(test_texts)

    y_pred = svm.predict(X_test)

    print(accuracy_score(test_labels, y_pred))