import os
import sys
import time
from random import randrange

import numpy as np

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from gensim.models import KeyedVectors

save_dir_path = os.path.join(sys.argv[4])
output_file = open(save_dir_path, "w+", encoding="utf-8")

MODEL_FILE_50 = os.path.join(sys.argv[2])
MODEL_FILE_300 = os.path.join(sys.argv[3])

word2vec_pre_trained_model_50 = None

word2vec_pre_trained_model_300 = None


def cross_validation(feature_vectors, class_vector):
    sm = SMOTE()
    features, labels = sm.fit_resample(feature_vectors, class_vector)
    classify = RandomForestClassifier()
    SKFold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)}

    results = cross_validate(estimator=classify, X=features, y=labels, cv=SKFold, scoring=scoring)
    output_file.writelines("Accuracy: " + '{:.4f}'.format(results['test_accuracy'].mean() * 100) + "\n")
    output_file.writelines("Precision: " + '{:.4f}'.format(results['test_precision'].mean() * 100) + "\n")
    output_file.writelines("Recall: " + '{:.4f}'.format(results['test_recall'].mean() * 100) + "\n")
    output_file.writelines("F1: " + '{:.4f}'.format(results['test_f1_score'].mean() * 100) + "\n")


# Create feature_vectors (the first argument of fit_resample in SMOTE)
# according to the last way of calculating words weights - choosing custom weights to words
def create_custom_feature_vectors(dataset, module, most_common_words):
    feature_vectors = []
    if module:
        model = word2vec_pre_trained_model_300
    else:
        model = word2vec_pre_trained_model_50
    for item in dataset:
        if not item:
            continue
        first_sentence = item[0].split(" ")
        second_sentence = item[1].split(" ")
        sum = 0
        both_sentences = first_sentence + second_sentence
        k = 0
        for word in both_sentences:
            if word in model.vocab:
                v_i = model.get_vector(word)
                if word in most_common_words:
                    sum += v_i * 0.3
                else:
                    if len(word) > 6:
                        sum += v_i * 10
                    else:
                        sum += v_i * 5
                k += 1
        # if sum is a vector of zeros, skip the current feature vector
        if not np.any(sum):
            continue
        feature_vector = sum / k
        feature_vectors.append(feature_vector)
    return feature_vectors


# Create feature_vectors (the first argument of fit_resample in SMOTE) according to the first two ways of calculating words weights
# module is 0 when we use 50d module and 1 when we use 300d module
# weight_type is 0 when we use w_i = 1 for all i and 1 when w_i in (0,1,2,3,4,5) for all i
def create_feature_vectors(dataset, module, weight_type):
    feature_vectors = []
    if module:
        model = word2vec_pre_trained_model_300
    else:
        model = word2vec_pre_trained_model_50
    for item in dataset:
        if not item:
            continue
        sum = 0
        k = 0
        for word in item[0].split(" ") + item[1].split(" "):
            if word in model.vocab:
                v_i = model.get_vector(word)
                # if weight_type == 1, do random weight, if == 0 do weight 1
                if weight_type == 1:
                    sum += (v_i * randrange(5))
                else:
                    sum += v_i
                k += 1
        if not np.any(sum):
            continue
        feature_vector = sum / k
        feature_vectors.append(feature_vector)
    return feature_vectors


def crete_dataset(en_dataset, simple_dataset, module, weight_type, most_common_words=None):
    if weight_type == 2:
        en_feature_vectors = create_custom_feature_vectors(en_dataset, module, most_common_words)
        simple_feature_vectors = create_custom_feature_vectors(simple_dataset, module, most_common_words)
    else:
        en_feature_vectors = create_feature_vectors(en_dataset, module, weight_type)
        simple_feature_vectors = create_feature_vectors(simple_dataset, module, weight_type)
    class_vector = [0] * len(en_feature_vectors) + [1] * len(simple_feature_vectors)
    dataset = en_feature_vectors + simple_feature_vectors
    return dataset, class_vector


def main():
    ## ################## A part 1 ####################
    ## Checking similarity between 'midfielder' and 'goalkeeper'
    # midfielder_goalkeeper_50 = word2vec_pre_trained_model_50.similarity('midfielder', 'goalkeeper')
    # midfielder_goalkeeper_300 = word2vec_pre_trained_model_300.similarity('midfielder', 'goalkeeper')
    #
    # print("Similirity between 'midfielder' and 'goalkeeper' 50d- " + str(midfielder_goalkeeper_50))
    # print("Similirity between 'midfielder' and 'goalkeeper' 300d- " + str(midfielder_goalkeeper_300))
    #
    # # Checking similarity between 'midfielder' and 'learning'
    # midfielder_learning_50 = word2vec_pre_trained_model_50.similarity('midfielder', 'learning')
    # midfielder_learning_300 = word2vec_pre_trained_model_300.similarity('midfielder', 'learning')
    #
    # print("Similirity between 'midfielder' and 'learning' 50d- " + str(midfielder_learning_50))
    # print("Similirity between 'midfielder' and 'learning' 300d- " + str(midfielder_learning_300))
    #
    # # Checking similarity between 'shallow' and 'deep'
    # shallow_deep_50 = word2vec_pre_trained_model_50.similarity('shallow', 'deep')
    # shallow_deep_300 = word2vec_pre_trained_model_300.similarity('shallow', 'deep')
    #
    # print("Similirity between 'shallow' and 'deep' 50d- " + str(shallow_deep_50))
    # print("Similirity between 'shallow' and 'deep' 300d- " + str(shallow_deep_300))
    #
    # # Checking similarity between 'shallow' and 'dog'
    # shallow_dog_50 = word2vec_pre_trained_model_50.similarity('shallow', 'dog')
    # shallow_dog_300 = word2vec_pre_trained_model_300.similarity('shallow', 'dog')
    #
    # print("Similirity between 'shallow' and 'dog' 50d- " + str(shallow_dog_50))
    # print("Similirity between 'shallow' and 'dog' 300d- " + str(shallow_dog_300))
    #
    # ################### A part 2 ####################
    # print("\n")
    # print("parrot")
    # print("\n#### 50d Model ####\n")
    # for item in word2vec_pre_trained_model_50.most_similar("parrot"):
    #     print(str(item[0]) + " " + str(item[1]))
    # print("\n#### 300d Model ####\n")
    # for item in word2vec_pre_trained_model_300.most_similar("parrot"):
    #     print(str(item[0]) + " " + str(item[1]))
    #
    # print("\n")
    # print("politician")
    # print("\n#### 50d Model ####\n")
    # for item in word2vec_pre_trained_model_50.most_similar("politician"):
    #     print(str(item[0]) + " " + str(item[1]))
    # print("\n#### 300d Model ####\n")
    # for item in word2vec_pre_trained_model_300.most_similar("politician"):
    #     print(str(item[0]) + " " + str(item[1]))
    #
    # print("\n")
    # print("journalist")
    # print("\n#### 50d Model ####\n")
    # for item in word2vec_pre_trained_model_50.most_similar("journalist"):
    #     print(str(item[0]) + " " + str(item[1]))
    # print("\n#### 300d Model ####\n")
    # for item in word2vec_pre_trained_model_300.most_similar("journalist"):
    #     print(str(item[0]) + " " + str(item[1]))
    #
    # print("\n")
    # print("cop")
    # print("\n#### 50d Model ####\n")
    # for item in word2vec_pre_trained_model_50.most_similar("cop"):
    #     print(str(item[0]) + " " + str(item[1]))
    # print("\n#### 300d Model ####\n")
    # for item in word2vec_pre_trained_model_300.most_similar("cop"):
    #     print(str(item[0]) + " " + str(item[1]))
    #
    # print(word2vec_pre_trained_model_50.most_similar(positive=['paris', 'country'], negative=['capital']))
    # print(word2vec_pre_trained_model_300.most_similar(positive=['paris', 'country'], negative=['capital']))
    #
    # print(word2vec_pre_trained_model_50.most_similar(positive=['louder', 'happy'], negative=['loud']))
    # print(word2vec_pre_trained_model_300.most_similar(positive=['louder', 'happy'], negative=['loud']))
    #
    # print(word2vec_pre_trained_model_50.most_similar(positive=['niece', 'man'], negative=['woman']))
    # print(word2vec_pre_trained_model_300.most_similar(positive=['niece', 'man'], negative=['woman']))
    input_dir = sys.argv[1]
    # Read all inputs files
    # English values will be kept in all_en_sentences
    # Simple English values will be kept in all_simple_sentences
    all_en_sentences = ""
    all_simple_sentences = ""
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), encoding='utf-8') as txtfile:
            lang = os.path.basename(txtfile.name).split("_")[0]
            content = txtfile.read()
            if lang == "en":
                all_en_sentences += content + "\n"
            else:
                all_simple_sentences += content + "\n"

    # Calculate most common words in the corpuses
    all_sentences = all_en_sentences + "\n" + all_simple_sentences
    words_list = Counter(all_sentences.replace("\n", " ").split(" "))
    most_common_words = words_list.most_common(100)

    # Take every two sentences in each corpus and make them into one- classifying unit of size 2 (chuck size = 2)
    # Filtering empty items in each 'filter'
    # Create a list of even indices and odd indices, then merge each item in index i of both lists into one new list
    even = list(filter(None, all_en_sentences.split("\n")))[0:][::2]
    odd = list(filter(None, all_en_sentences.split("\n")))[1:][::2]
    en_content = list(map(list, zip(even, odd)))

    even = list(filter(None, all_simple_sentences.split("\n")))[0:][::2]
    odd = list(filter(None, all_simple_sentences.split("\n")))[1:][::2]
    simple_content = list(map(list, zip(even, odd)))

    output_file.writelines("Arithmetic mean:\n")
    output_file.writelines("word2vec_50 model performance:\n")
    dataset, class_vector = crete_dataset(en_content, simple_content, 0, 0)
    cross_validation(dataset, class_vector)
    output_file.writelines("word2vec_300 model performance:\n")
    dataset, class_vector = crete_dataset(en_content, simple_content, 0, 1)
    cross_validation(dataset, class_vector)
    output_file.writelines(
        "-------------------------------------------------------------------------------------------------------------------\n")
    output_file.writelines("Random weights:\n")
    output_file.writelines("word2vec_50 model performance:\n")
    dataset, class_vector = crete_dataset(en_content, simple_content, 1, 0)
    cross_validation(dataset, class_vector)
    output_file.writelines("word2vec_300 model performance:\n")
    dataset, class_vector = crete_dataset(en_content, simple_content, 1, 1)
    cross_validation(dataset, class_vector)
    output_file.writelines(
        "-------------------------------------------------------------------------------------------------------------------\n")
    output_file.writelines("My weights:\n")
    output_file.writelines("word2vec_50 model performance:\n")
    dataset, class_vector = crete_dataset(en_content, simple_content, 0, 2, most_common_words)
    cross_validation(dataset, class_vector)
    output_file.writelines("word2vec_300 model performance:\n")
    dataset, class_vector = crete_dataset(en_content, simple_content, 1, 2, most_common_words)
    cross_validation(dataset, class_vector)


if __name__ == "__main__":
    start_time = time.time()
    word2vec_pre_trained_model_50 = KeyedVectors.load_word2vec_format(MODEL_FILE_50, binary=False)
    word2vec_pre_trained_model_300 = KeyedVectors.load_word2vec_format(MODEL_FILE_300, binary=False)
    main()
    print("The run has taken %s seconds" % np.math.floor((time.time() - start_time)))
