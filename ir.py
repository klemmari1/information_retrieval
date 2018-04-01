from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import csv

with open('scores.csv', 'w', newline='') as csvfile:
    scorewriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    scorewriter.writerow(['Query',
                          'Binary euclidean precision', 'Binary euclidean recall', 'Binary euclidean F-score',
                          'Binary cosine precision', 'Binary cosine recall', 'Binary cosine F-score',
                          'TF euclidean precision', 'TF euclidean recall', 'TF euclidean F-score',
                          'TF cosine precision', 'TF cosine recall', 'TF cosine F-score',
                          'TF-IDF euclidean precision', 'TF-IDF euclidean recall', 'TF-IDF euclidean F-score',
                          'TF-IDF cosine precision', 'TF-IDF cosine recall', 'TF-IDF cosine F-score'])

    for q in range(1, 226):
        print("QUERY " + str(q) + ":")
        # prepare corpus and relevance
        corpus = []
        relevance = ""
        for d in range(1400):
            f = open("./d/"+str(d+1)+".txt")
            corpus.append(f.read())
        # add query to corpus
        f = open("./q/"+str(q)+".txt")
        corpus.append(f.read())
        # relevance
        f = open("./r/"+str(q)+".txt")
        relevance = np.array(list(map(int, filter(None, f.read().split("\n")))))

        # init vectorizers
        count_vectorizer = CountVectorizer()
        tf_vectorizer = TfidfVectorizer(use_idf=False)
        tfidf_vectorizer = TfidfVectorizer()

        # prepare matrices
        count_matrix = count_vectorizer.fit_transform(corpus)
        tf_matrix = tf_vectorizer.fit_transform(corpus)
        tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

        # compute similarity between query and all docs and get top 10 relevant
        esim = np.array(euclidean_distances(count_matrix[len(corpus)-1], count_matrix[0:(len(corpus)-1)])[0])
        #etopRelevant = esim.argsort()[:10]+1
        csim = np.array(cosine_similarity(count_matrix[len(corpus)-1], count_matrix[0:(len(corpus)-1)])[0])
        #ctopRelevant = csim.argsort()[-10:][::-1]+1
        #print("Top Binary representation euclidean distances: " + str(etopRelevant))
        #print("Top Binary representation cosine similarity: " + str(ctopRelevant))
        escores1 = precision_recall_fscore_support(relevance, esim.argsort()[:len(relevance)]+1, average='micro')
        cscores1 = precision_recall_fscore_support(relevance, csim.argsort()[-len(relevance):][::-1]+1, average='micro')
        print("Binary euclidean scores: Precision: " + str(escores1[0]) + ", Recall: " + str(escores1[1]) + ", F-Score: " + str(escores1[2]))
        print("Binary cosine scores: Precision: " + str(cscores1[0]) + ", Recall: " + str(cscores1[1]) + ", F-Score: " + str(cscores1[2]))

        print("\n")

        esim = np.array(euclidean_distances(tf_matrix[len(corpus)-1], tf_matrix[0:(len(corpus)-1)])[0])
        #etopRelevant = esim.argsort()[:10]+1
        csim = np.array(cosine_similarity(tf_matrix[len(corpus)-1], tf_matrix[0:(len(corpus)-1)])[0])
        #ctopRelevant = csim.argsort()[-10:][::-1]+1
        #print("Top Term frequency euclidean distances: " + str(etopRelevant))
        #print("Top Term frequency cosine similarity: " + str(ctopRelevant))
        escores2 = precision_recall_fscore_support(relevance, esim.argsort()[:len(relevance)]+1, average='micro')
        cscores2 = precision_recall_fscore_support(relevance, csim.argsort()[-len(relevance):][::-1]+1, average='micro')
        print("Term frequency euclidean scores: Precision: " + str(escores2[0]) + ", Recall: " + str(escores2[1]) + ", F-Score: " + str(escores2[2]))
        print("Term frequency cosine scores: Precision: " + str(cscores2[0]) + ", Recall: " + str(cscores2[1]) + ", F-Score: " + str(cscores2[2]))

        print("\n")

        esim = np.array(euclidean_distances(tfidf_matrix[len(corpus)-1], tfidf_matrix[0:(len(corpus)-1)])[0])
        #etopRelevant = esim.argsort()[:10]+1
        csim = np.array(cosine_similarity(tfidf_matrix[len(corpus)-1], tfidf_matrix[0:(len(corpus)-1)])[0])
        #ctopRelevant = csim.argsort()[-10:][::-1]+1
        #print("Top TF-IDF euclidean distances: " + str(etopRelevant))
        #print("Top TF-IDF cosine similarity: " + str(ctopRelevant))
        escores3 = precision_recall_fscore_support(relevance, esim.argsort()[:len(relevance)]+1, average='micro')
        cscores3 = precision_recall_fscore_support(relevance, csim.argsort()[-len(relevance):][::-1]+1, average='micro')
        print("TF-IDF euclidean scores: Precision: " + str(escores3[0]) + ", Recall: " + str(escores3[1]) + ", F-Score: " + str(escores3[2]))
        print("TF-IDF cosine scores: Precision: " + str(cscores3[0]) + ", Recall: " + str(cscores3[1]) + ", F-Score: " + str(cscores3[2]))

        print("\n")
        print("\n")

        scorewriter.writerow([q,
        escores1[0], escores1[1], escores1[2], cscores1[0], cscores1[1], cscores1[2],
        escores2[0], escores2[1], escores2[2], cscores2[0], cscores2[1], cscores2[2],
        escores3[0], escores3[1], escores3[2], cscores3[0], cscores3[1], cscores3[2]])
