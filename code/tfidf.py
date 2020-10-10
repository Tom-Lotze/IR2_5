# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2020-10-09 17:28
# @Last Modified by:   TomLotze
# @Last Modified time: 2020-10-10 18:20


from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer()

queries = ["Query number one", 'query number two']
questions = ["Question number one clarification", 'question number monkey two']
answers = ["answer number one", "answer number two","answer number three","answer number four", "answer number five", "answer number six", "answer number seven","answer number eight","answer number nine", "answer number ten"]


# zip(question)

corpus = [" ".join([queries[i], questions[i], " ".join(answers[i*5:i*5+5])]) for i in range(len(queries))]
# every element in corpus (list of strings should correspond to an instance:
# words in query, question and all answers)


print(corpus)

X = vectorizer.fit_transform(corpus)

# X is here a sparse matrix of n_samples x nr_features (=instances x vocab size)

print(X)
breakpoint()
