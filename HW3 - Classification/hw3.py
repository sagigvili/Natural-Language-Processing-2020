import math
import random
import re
import sys
import os.path
import time
from statistics import mean as average
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

save_dir_path = os.path.join(sys.argv[3])
output_file = open(save_dir_path + "/hw3_output.txt", "w+", encoding="utf-8")


def classifying(trained_set, class_vector):
    text_clf_base = MultinomialNB()
    text_clf_base.fit(trained_set, class_vector)

    text_clf_knn = KNeighborsClassifier()
    text_clf_knn.fit(trained_set, class_vector)

    text_clf_logistic = LogisticRegression(solver='liblinear')
    text_clf_logistic.fit(trained_set, class_vector)

    output_file.writelines("Naïve Bayes:")
    output_file.writelines('{:.2f}'.format(cross_val_score(text_clf_base, trained_set, class_vector, cv=10).mean() * 100))
    output_file.writelines("\n")
    output_file.writelines("KNN:")
    output_file.writelines('{:.2f}'.format(cross_val_score(text_clf_knn, trained_set, class_vector, cv=10).mean() * 100))
    output_file.writelines("\n")
    output_file.writelines("Logistic Regression:")
    output_file.writelines('{:.2f}'.format(cross_val_score(text_clf_logistic, trained_set, class_vector, cv=10).mean() * 100))
    output_file.writelines("\n")


# Custom features options, need at least 15:
#   Sentence length, punctuation, unique phrases, average_word length
def custom_features(mixed_corpora, class_vector):
    training_set = mixed_corpora[0] + mixed_corpora[1]
    features_vector = []
    for sentence in training_set:
        splited_sentence = sentence.split(" ")
        vector = [len(sentence.split(" "))]  # Sentence length
        vector.append(avg_word_length(splited_sentence))  # Average word length
        vector.append(count_comas(splited_sentence))  # Number of comas
        vector.append(count_numbers(splited_sentence))  # Number of numbers in the sentence (like "1918")
        vector.append(count_ing(splited_sentence))  # Number of words ends with "ing"
        vector.append(different_words(sentence))  # Number of different words in sentence (normalized)
        vector.append(count_pattern(sentence,
                                    "and"))  # each use of count_pattern in the next lines counts the occurrences of pattern in sentence
        vector.append(count_pattern(sentence, "of"))
        vector.append(count_pattern(sentence, "in"))
        vector.append(count_pattern(sentence, "the"))
        vector.append(count_pattern(sentence, "with"))
        vector.append(count_openers(splited_sentence))
        vector.append(count_closers(splited_sentence))
        #### Content props ####
        vector.append(count_pattern(sentence, "coffee"))
        vector.append(count_pattern(sentence, "system"))
        vector.append(count_pattern(sentence, "basketball"))
        vector.append(count_pattern(sentence, "leonardo"))
        vector.append(count_pattern(sentence, "people"))
        features_vector.append(vector)

    best_features = SelectKBest(k=15)
    best_15_results = best_features.fit_transform(features_vector, class_vector)
    ##### Uncomment lines 74-95 to see the scores of each feature
    # scores = {}
    # scores['s_length'] = best_features.scores_[0]
    # scores['avg_word_len'] = best_features.scores_[1]
    # scores['comas'] = best_features.scores_[2]
    # scores['numbers'] = best_features.scores_[3]
    # scores['ends_ing'] = best_features.scores_[4]
    # scores['diffrent_words'] = best_features.scores_[5]
    # scores['and'] = best_features.scores_[6]
    # scores['of'] = best_features.scores_[7]
    # scores['in'] = best_features.scores_[8]
    # scores['the'] = best_features.scores_[9]
    # scores['with'] = best_features.scores_[10]
    # scores['openers'] = best_features.scores_[11]
    # scores['closers'] = best_features.scores_[12]
    # scores['coffee'] = best_features.scores_[13]
    # scores['system'] = best_features.scores_[14]
    # scores['basketball'] = best_features.scores_[15]
    # scores['leonardo'] = best_features.scores_[16]
    # scores['people'] = best_features.scores_[17]
    # scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    # print(scores)
    # print(best_features.scores_[best_features.get_support()])

    classifying(best_15_results, class_vector)


def avg_word_length(sentence):
    words_len_vector = []
    for word in sentence:
        words_len_vector.append(len(word))
    return average(words_len_vector)


def different_words(sentence):
    tokenized_sentence = nltk.tokenize.word_tokenize(sentence)
    return len(nltk.Counter(tokenized_sentence).keys()) / len(sentence.split(" "))


def count_comas(sentence):
    counter = 0
    for word in sentence:
        if re.match(r'\,', word):
            counter += 1
    return counter


def count_openers(sentence):
    counter = 0
    for word in sentence:
        if re.match(r'\[|\(\{', word):
            counter += 1
    return counter


def count_closers(sentence):
    counter = 0
    for word in sentence:
        if re.match(r'\]|\)|\}', word):
            counter += 1
    return counter


def count_upper_case(sentence):
    counter = 0
    for word in sentence:
        if word.isupper():
            counter += 1
    return counter


def count_numbers(sentence):
    counter = 0
    for word in sentence:
        if word.isdigit():
            counter += 1
    return counter


def count_initials(sentence):
    counter = 0
    for word in sentence:
        if word.isupper() and word.isalpha():
            counter += 1
    return counter


def count_ing(sentence):
    counter = 0
    for word in sentence:
        if word.endswith("ing"):
            counter += 1
    return counter


def count_pattern(sentence, pattern):
    return len(re.findall(" " + pattern + " ", sentence.lower()))


def top_300(mixed_corpora, class_vector):
    training_set = mixed_corpora[0] + mixed_corpora[1]
    top_300_words = os.path.join(sys.argv[2])
    with open(top_300_words, encoding='utf-8') as file:
        top_300_words = file.readlines()
        top_300_words = [word.replace(' \n', '') for word in top_300_words]
    feature_vectors = []

    # Create 300th length vector for each sentence
    for sentence in training_set:
        boolean_vector = []
        for word in top_300_words:
            if word in sentence:
                boolean_vector.append(1)
            else:
                boolean_vector.append(0)
        feature_vectors.append(boolean_vector)
    t = TfidfTransformer()
    tfid = t.fit_transform(feature_vectors, class_vector)
    classifying(tfid, class_vector)


def bag_of_words(mixed_corpora, class_vector):
    training_set = mixed_corpora[0] + mixed_corpora[1]
    vectorizer = CountVectorizer()
    vectorized = vectorizer.fit_transform(training_set)
    t = TfidfTransformer()
    tfid = t.fit_transform(vectorized, class_vector)
    classifying(tfid, class_vector)


def main():
    input_dir = sys.argv[1]
    en_values_corpora = {}
    simple_values_corpora = {}
    # We'll go through all files in the input directory and add each text file content to
    # "finished corpora" in its key (which is the language the file is written in)
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), encoding='utf-8') as txtfile:
            lang = os.path.basename(txtfile.name).split("_")[0]
            name = os.path.basename(txtfile.name).split("_")[1].replace(".txt", "")
            if lang == "en":
                en_values_corpora[name] = txtfile.read().split("\n")
            else:
                simple_values_corpora[name] = txtfile.read().split("\n")

            # if value has been retrieved in both languages- equal their number of sentences
            # by randomizing k sentences from the longer one, where k the is the number of sentences
            # in the shorter value
            if name in en_values_corpora.keys() and name in simple_values_corpora.keys():
                en_sentences_count = len(en_values_corpora[name])
                simple_sentences_count = len(simple_values_corpora[name])
                if en_sentences_count > simple_sentences_count:
                    en_values_corpora[name] = random.sample(en_values_corpora[name], simple_sentences_count)
                elif en_sentences_count < simple_sentences_count:
                    simple_values_corpora[name] = random.sample(simple_values_corpora[name], en_sentences_count)

    en_corpora = []
    simple_corpora = []
    for key in en_values_corpora:
        for item in en_values_corpora[key]:
            en_corpora.append(item)
        for item in simple_values_corpora[key]:
            simple_corpora.append(item)

    ####### Uncomment lines 239-266 for printing the number of diffrent words in each corpus
    ####### And the appearnces of each content feature on each corpus
    # words_en = nltk.tokenize.word_tokenize("\n".join(en_corpora))
    # word_counter_en = nltk.Counter(words_en)
    # print("coffee")
    # print(word_counter_en['coffee'])
    # print("system")
    # print(word_counter_en['system'])
    # print("basketball")
    # print(word_counter_en['basketball'])
    # print("Leonardo")
    # print(word_counter_en['Leonardo'])
    # print("people")
    # print(word_counter_en['people'])

    # words_simple = nltk.tokenize.word_tokenize("\n".join(simple_corpora))
    # word_counter_simple = nltk.Counter(words_simple)
    # print("coffee")
    # print(word_counter_simple['coffee'])
    # print("system")
    # print(word_counter_simple['system'])
    # print("basketball")
    # print(word_counter_simple['basketball'])
    # print("Leonardo")
    # print(word_counter_simple['Leonardo'])
    # print("people")
    # print(word_counter_simple['people'])

    # print("Number of different en words: ", len(list(word_counter_en.keys())))
    # print("Number of different simple words: ", len(list(word_counter_simple.keys())))

    # Assign a classifying number to each item in the features vector
    # 0 for simple english class, 1 for standard
    mixed_corpora = {0: simple_corpora, 1: en_corpora}
    class_vector_0 = [0] * len(mixed_corpora[0])
    class_vector_1 = [1] * len(mixed_corpora[1])
    class_vector = class_vector_0 + class_vector_1

    output_file.writelines("Phase1 (Bag of Words):\n")
    bag_of_words(mixed_corpora, class_vector)
    output_file.writelines(
        "-------------------------------------------------------------------------------------------------------------------")
    output_file.writelines("\nPhase2 (300 most frequent words):\n")
    top_300(mixed_corpora, class_vector)
    output_file.writelines(
        "-------------------------------------------------------------------------------------------------------------------")
    output_file.writelines("\nPhase3 (My features):\n")
    custom_features(mixed_corpora, class_vector)

    output_file.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("The run has taken %s seconds" % math.floor((time.time() - start_time)))
