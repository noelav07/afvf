# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import random
import re
import pickle5 as pickle
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
import re
from nltk.corpus import wordnet

dataset = pd.read_csv("review.csv", sep="\t")

import pandas as pd
import random
import re
import pickle5 as pickle
import nltk
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.datasets import load_files

with open("Pickle Files/classifier.pickle", "rb") as f:
    clf = pickle.load(f)

with open("Pickle Files/TfidfModel.pickle", "rb") as f:
    tfidf = pickle.load(f)


def getSentiment(text):
    text = str(text).lower()
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"who're", "who are", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"\s+[a-z]\s+", " ", text)
    text = re.sub(r"^[a-z]\s+", " ", text)
    text = re.sub(r"\s+[a-z]$", " ", text)
    text = re.sub(r"\s+", " ", text)
    sent = clf.predict(tfidf.transform([text]).toarray())
    return sent[0]


remove_reviews = []

for i in range(len(dataset)):
    if getSentiment(dataset["review_headline"][i]) != getSentiment(dataset["review_body"][i]):
        remove_reviews.append(dataset["review_id"][i])

print(remove_reviews)

import pandas as pd
dataset = pd.read_csv("reviews.csv", sep="\t")

customers = dataset.groupby("customer_id")
customer_list = dataset["customer_id"].unique()
size = len(customer_list.tolist())

for i in range(size):
    brand_df = customers.get_group(customer_list[i])
    brands = brand_df.groupby("product_parent")
    brands_list = brand_df["product_parent"].unique()
    no_of_brands = len(brands_list.tolist())

    for j in range(no_of_brands):
        product_df = brands.get_group(brands_list[j])
        no_of_products = len(product_df["product_id"])
        if no_of_products <= 2:
            continue
        indices = product_df.index.values.tolist()
        sentiment = getSentiment(product_df["review_body"][indices[0]])
        isSameSentiment = True
        if no_of_products < 4:
            continue
        for k in range(1, no_of_products):
            text = str(product_df["review_body"][indices[k]])
            if getSentiment(text) != sentiment:
                isSameSentiment = False
                break
        if isSameSentiment:
            remove_reviews.append(customer_list[i])
            break

print(remove_reviews)

ip = dataset.groupby("IP Address")
ip_list = dataset["IP Address"].unique()
remove_ip = []
size = len(ip_list.tolist())
for i in range(size):
    brand_df = ip.get_group(ip_list[i])
    brands = brand_df.groupby("product_parent")
    brands_list = brand_df["product_parent"].unique()
    no_of_brands = len(brands_list.tolist())
    for j in range(no_of_brands):
        product_df = brands.get_group(brands_list[j])
        no_of_products = len(product_df["product_id"])
        if no_of_products <= 2:
            break
        indices = product_df.index.tolist()
        sentiment = getSentiment(product_df["review_body"][indices[0]])
        isSameSentiment = True
        for k in range(1, no_of_products):
            text = str(product_df["review_body"][indices[k]])
            if getSentiment(text) != sentiment:
                isSameSentiment = False
                break
        if isSameSentiment:
            remove_ip.append(ip_list[i])

remove_ip
dataset.sort_values("customer_id", inplace=True)
remove_reviews = []
customer_group = dataset.groupby("customer_id")
customer_group_list = dataset["customer_id"].unique().tolist()
for i in range(len(customer_group_list)):
    customer_reviews = customer_group.get_group(customer_group_list[i])
    dates_list = customer_reviews["review_date"].dropna().unique().tolist()
    reviews_by_date = customer_reviews.groupby("review_date")

    for j in range(len(dates_list)):
        reviews_by_date_for_pos = []
        reviews_by_date_for_neg = []

        df = reviews_by_date.get_group(dates_list[j])
        indices = df.index.tolist()

        for k in range(len(df)):
            text = df["review_body"][indices[k]]
            if getSentiment(text) == 0:
                reviews_by_date_for_neg.append(df["review_id"][indices[k]])
            else:
                reviews_by_date_for_pos.append(df["review_id"][indices[k]])

        if len(reviews_by_date_for_pos) > 3:
            remove_reviews.extend(reviews_by_date_for_pos)

        if len(reviews_by_date_for_neg) > 3:
            remove_reviews.extend(reviews_by_date_for_neg)
len(remove_reviews)
print(remove_reviews)
remove_reviews = []
ip_group = dataset.groupby("IP Address")
ip_list = dataset["IP Address"].unique().tolist()

for i in range(len(ip_list)):
    reviews = ip_group.get_group(ip_list[i])
    dates_list = reviews["review_date"].dropna().unique().tolist()
    reviews_by_date = reviews.groupby("review_date")

    for j in range(len(dates_list)):
        reviews_by_date_for_pos = []
        reviews_by_date_for_neg = []

        try:
            reviews_for_each_day = reviews_by_date.get_group(dates_list[j])
        except KeyError:
            continue

        indices = reviews_for_each_day.index.tolist()

        for k in range(len(reviews_for_each_day)):
            text = reviews_for_each_day["review_body"][indices[k]]
            if getSentiment(text) == 0:
                reviews_by_date_for_neg.append(reviews_for_each_day["review_id"][indices[k]])
            else:
                reviews_by_date_for_pos.append(reviews_for_each_day["review_id"][indices[k]])

        if len(reviews_by_date_for_pos) > 3:
            remove_reviews.extend(reviews_by_date_for_pos)

        if len(reviews_by_date_for_neg) > 3:
            remove_reviews.extend(reviews_by_date_for_neg)

len(remove_reviews)
print(remove_reviews)

tfidf_vectorizer = TfidfVectorizer()

dataset.reset_index()
dataset.set_index("review_id")
dataset.sort_values("timestamp", inplace=True)


def OnlyStopwords(str):
    words = nltk.word_tokenize(str)
    words = [word for word in words if word not in stopwords.words("english")]
    if len(words) == 0:
        return True
    return False


from nltk.corpus import wordnet

remove_reviews = []
indices = []
for i in range(len(dataset)):

    reviews = [str(dataset["review_body"][i])]

    try:
        tfidf_vectorizer.fit_transform(reviews)
    except:
        remove_reviews.append(dataset["review_id"][i])
        continue

    Time = dataset["timestamp"][i]

    for j in range(i + 1, len(dataset)):

        indices.append(dataset["review_id"][j])

        if dataset["timestamp"][j] - Time <= 1800:
            reviews.append(str(dataset["review_body"][j]))
        else:
            break

    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

    tfidf_list = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).tolist()

    i_appended = False
    for k in range(1, len(tfidf_list[0])):

        if tfidf_list[0][k] > 0.6:
            remove_reviews.append(dataset["review_id"][i + k])

            if not i_appended:
                remove_reviews.append(dataset["review_id"][i])
                i_appended = True

len(remove_reviews)

print(remove_reviews)

for i in range(len(dataset)):
    words = nltk.word_tokenize(str(dataset["review_body"][i]))
    tagged_words = nltk.pos_tag(words)
    nouns_count = 0
    verbs_count = 0

    for j in range(len(tagged_words)):
        if tagged_words[j][1].startswith("NN"):
            nouns_count += 1

        if tagged_words[j][1].startswith("VB"):
            verbs_count += 1

    if verbs_count > nouns_count:
        remove_reviews.append(dataset["review_id"][i])

print(remove_reviews)

len(remove_reviews)

for i in range(len(dataset)):
    dataset["review_body"][i] = str(dataset["review_body"][i]).lower()
    words = nltk.word_tokenize(dataset["review_body"][i])
    sentence = nltk.sent_tokenize(dataset["review_body"][i])
    count = 0
    if len(sentence) > 4:
        for j in range(len(words)):
            if words[j] == "i" or words[j] == "we":
                count += 1
        if count > len(sentence) / 2:
            remove_reviews.append(dataset["review_id"][i])

len(remove_reviews)

print(remove_reviews)

dataset.set_index("review_id", inplace=True)


def LSA(text):

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text)

    lsa = TruncatedSVD(n_components=1, n_iter=100)
    lsa.fit(X)

    terms = vectorizer.get_feature_names()
    concept_words = {}

    for j, comp in enumerate(lsa.components_):
        componentTerms = zip(terms, comp)
        sortedTerms = sorted(componentTerms, key=lambda x: x[1], reverse=True)
        sortedTerms = sortedTerms[:10]
        concept_words[str(j)] = sortedTerms

    sentence_scores = []
    for key in concept_words.keys():
        for sentence in text:
            words = nltk.word_tokenize(sentence)
            scores = 0
            for word in words:
                for word_with_scores in concept_words[key]:
                    if word == word_with_scores[0]:
                        scores += word_with_scores[1]
            sentence_scores.append(scores)
    return sentence_scores


import nltk
nltk.download('omw-1.4')
product_df = dataset.groupby("product_id")

unique_products = dataset["product_id"].unique()
no_products = len(unique_products.tolist())
remove_reviews = []

for i in range(no_products):
    df = product_df.get_group(unique_products[i])
    unique_reviews = df.index.tolist()
    no_reviews = len(unique_reviews)
    count = no_reviews
    reviews = []
    review_id = []
    for j in range(no_reviews):
        text = str(df.loc[unique_reviews[j]]["review_body"])
        text = re.sub(r"\W", " ", text)
        text = re.sub(r"\d", " ", text)
        text = re.sub(r"\s+[a-z]\s+", " ", text)
        text = re.sub(r"^[a-z]\s+", " ", text)
        text = re.sub(r"\s+[a-z]$", " ", text)
        text = re.sub(r"\s+", " ", text)
        words = nltk.word_tokenize(text)
        if len(words) == 1:
            if len(text) <= 1 or not wordnet.synsets(text):
                remove_reviews.append(unique_reviews[j])
                count -= 1
                continue
        elif len(words) == 0:
            remove_reviews.append(unique_reviews[j])
            count -= 1
            continue
        review_id.append(unique_reviews[j])
        reviews.append(text)
    if count <= 0:
        continue
    if count == 1:
        text = [text, str(df.loc[review_id[0]]["product_title"])]
        sentence_scores = LSA(text)
        if sentence_scores[0] == 0:
            remove_reviews.append(review_id[0])
        continue
    sentence_scores = LSA(reviews)
    for j in range(len(sentence_scores)):
        if sentence_scores[j] == 0.00:
            remove_reviews.append(review_id[j])

len(remove_reviews)
print(remove_reviews)

dataset.drop(remove_reviews, inplace=True)
dataset = dataset.set_index("IP Address")
dataset.to_csv("real_reviews.csv", sep="\t")

