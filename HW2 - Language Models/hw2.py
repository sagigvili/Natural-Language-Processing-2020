import random
import sys
import os.path
import time
import math

save_dir_path = os.path.join(sys.argv[2])
source_file = open(save_dir_path + "/hw2_output.txt", "w+", encoding="utf-8")


def random_bigrams_sentences(probs, random_length):
    sentence = ""
    starters_list = {}
    # Make a list of words after <s> and their probability
    for key in probs:
        key = key.split(" ")
        if key[0] == "<s>":
            starters_list[key[1]] = probs[" ".join(key)]
    random_pair = random.choices(list(starters_list.keys()), list(starters_list.values()), k=1)
    length = 0
    # Using Shannon's method to create a random sentence using biagram model, until we get to random length of words or "</s>"
    while random_pair[0] != "</s>" or length < random_length[0]:
        if random_pair[0] not in ["<s>", "</s>"]:
            sentence += random_pair[0] + " "
            length += 1
        # Given the current first token, we'll create a list if all possible tokens coming afterwards
        next_word_list = {}
        for key in probs:
            if key.split(" ")[0] == random_pair[0]:
                list_key = key.split(" ")[1]
                next_word_list[list_key] = probs[key]

        # Handle situations where next word list is empty - we start to random words again from starters list
        if next_word_list:
            random_pair = random.choices(list(next_word_list.keys()), list(next_word_list.values()), k=1)
        else:
            random_pair = random.choices(list(starters_list.keys()), list(starters_list.values()), k=1)

    return sentence


def random_trigrams_sentences(probs, random_length):
    sentence = ""
    starters_list = {}
    # Make a list of words after <s> and their probability- starters list
    for key in probs:
        key = key.split(" ")
        if key[0] == "<s>":
            key = " ".join(key)
            starters_list[key] = probs[key]
    random_triple = random.choices(list(starters_list.keys()), list(starters_list.values()), k=1)
    triple = random_triple[0].split(" ")
    sentence += triple[1] + " "
    length = 0
    # Using Shannon's method to create a random sentence using triagram model, until we get to random length of words or "</s>"
    while triple[2] != "</s>" or length < random_length[0]:
        if triple[2] not in ["<s>", "</s>"]:
            sentence += triple[2] + " "
            length += 1
        # Given the current first triple, we'll create a list if all possible tokens coming afterwards
        next_word_list = {}
        for key in probs:
            key_splited = key.split(" ")
            if key_splited[0] == triple[-2] and key_splited[1] == triple[-1]:
                list_key = key_splited[2]
                next_word_list[list_key] = probs[key]
        # Handle situations where next word list is empty - we start to random words again from starters list
        if next_word_list:
            random_word = random.choices(list(next_word_list.keys()), list(next_word_list.values()), k=1)
        else:
            random_word = random.choices(list(starters_list.keys()), list(starters_list.values()), k=1)
        triple = [triple[1], triple[2], random_word[0]]
    return sentence


# Generate random sentences, num indicates whether it is by unigrams - 1, bigrams - 2 or trigrams - 3 module
def random_sentences(corpora, num):
    for lang in corpora.keys():
        if lang == "en":
            if num == 1:
                source_file.writelines("Unigrams model based on complete dataset (English):" + "\n\n")
            if num == 2:
                source_file.writelines("\n")
                source_file.writelines("Bigrams model based on complete dataset (English):" + "\n\n")
            if num == 3:
                source_file.writelines("\n")
                source_file.writelines("Trigrams model based on complete dataset (English):" + "\n\n")
        elif lang == "es":
            if num == 1:
                source_file.writelines("\n")
                source_file.writelines("Unigrams model based on complete dataset (Spanish):" + "\n\n")
            if num == 2:
                source_file.writelines("\n")
                source_file.writelines("Bigrams model based on complete dataset (Spanish):" + "\n\n")
            if num == 3:
                source_file.writelines("\n")
                source_file.writelines("Trigrams model based on complete dataset (Spanish):" + "\n\n")
        elif lang == "simple":
            if num == 1:
                source_file.writelines("\n")
                source_file.writelines("Unigrams model based on complete dataset (simple English):" + "\n\n")
            if num == 2:
                source_file.writelines("\n")
                source_file.writelines("Bigrams model based on complete dataset (simple English):" + "\n\n")
            if num == 3:
                source_file.writelines("\n")
                source_file.writelines("Trigrams model based on complete dataset (simple English):" + "\n\n")
        for i in range(0, 3):
            # Pick random length of sentence out if the lengths distribution
            random_length = random.choices(list(corpora[lang][0][3].keys()), list(corpora[lang][0][3].values()), k=1)
            sentence = ""
            # Unigrams
            if num == 1:
                # We'll choose random unigrams from the corpus (filtering <s> and </s>) until we get the random length of words
                for i in range(0, random_length[0]):
                    random_word = random.choices(list(corpora[lang][num][1].keys()),
                                                 list(corpora[lang][num][1].values()),
                                                 k=1)
                    while random_word[0] in ["<s>", "</s>"]:
                        random_word = random.choices(list(corpora[lang][num][1].keys()),
                                                     list(corpora[lang][num][1].values()),
                                                     k=1)
                    sentence += random_word[0] + " "
            if num == 2:
                sentence = random_bigrams_sentences(corpora[lang][num][1], random_length)
            # Trigrams
            if num == 3:
                sentence = random_trigrams_sentences(corpora[lang][num][1], random_length)
            source_file.writelines(sentence + "\n")


def random_phrase(corpora, num_of_tokens):
    content = ""
    # Create a string of all 3 corpuses content
    for key in corpora:
        content += corpora[key][0][0]
    content = content.split(" ")
    corpus_size = len(content)
    index = random.randrange(corpus_size - 1)
    if num_of_tokens > 1:
        # if the index is at the end of the corpus, random again
        while index >= corpus_size - num_of_tokens:
            index = random.randrange(corpus_size - 1)
        return " ".join(content[index:index + num_of_tokens])
    else:
        return content[index]


def trigrams_phrases(corpora_with_probs):
    # Sample 3,4,5 and 7 words phrases
    s_list = [random_phrase(corpora_with_probs, 3), "feel no fear", random_phrase(corpora_with_probs, 4),
              "los angeles lakers team",
              random_phrase(corpora_with_probs, 5),
              "Death of a Ladies is", random_phrase(corpora_with_probs, 7),
              "Montreal , Quebec , Israel , Canada"]

    # Get each corpus' unigram, bigrams and trigrams probs table, trigram types, pair types, token types and size
    tokens_probs_en = corpora_with_probs['en'][1][1]
    pairs_probs_en = corpora_with_probs['en'][2][1]
    triples_probs_en = corpora_with_probs['en'][3][1]
    corpus_size_en = len(corpora_with_probs['en'][0][0].split(" "))
    tokens_types_en = len(corpora_with_probs['en'][1][0].keys())
    pairs_types_en = len(corpora_with_probs['en'][2][0].keys())
    triples_types_en = len(corpora_with_probs['en'][3][0].keys())

    tokens_probs_es = corpora_with_probs['es'][1][1]
    pairs_probs_es = corpora_with_probs['es'][2][1]
    triples_probs_es = corpora_with_probs['es'][3][1]
    corpus_size_es = len(corpora_with_probs['es'][0][0].split(" "))
    tokens_types_es = len(corpora_with_probs['es'][1][0].keys())
    pairs_types_es = len(corpora_with_probs['es'][2][0].keys())
    triples_types_es = len(corpora_with_probs['es'][3][0].keys())

    tokens_probs_simple = corpora_with_probs['simple'][1][1]
    pairs_probs_simple = corpora_with_probs['simple'][2][1]
    triples_probs_simple = corpora_with_probs['simple'][3][1]
    corpus_size_simple = len(corpora_with_probs['simple'][0][0].split(" "))
    tokens_types_simple = len(corpora_with_probs['simple'][1][0].keys())
    pairs_types_simple = len(corpora_with_probs['simple'][2][0].keys())
    triples_types_simple = len(corpora_with_probs['simple'][3][0].keys())

    delta_a = 0.6
    delta_b = 0.2
    delta_c = 0.2

    # Decide what corpus each phrase is from
    # If "w_n-1" is not found in corpus- smooth it using (C(w_n-1) + 1) / N + V formula where C(w_n-1) = 0
    # If "w_n-1w_n" is not found in corpus- smooth it using (C(w_n-1w_n) + 1) / (C(w_n-1) + V) where C(w_n-1w_n) = 0 and C(w_n) = 0
    # If "w_n-2w_n-1w_n" is not found in corpus- smooth it using (C(w_n-2w_n-1w_n) + 1) / (C(w_n-1w_n) + V) where C(w_n-2w_n-1w_n) = 0 and C(w_n-1w_n) = 0
    # then linear backoff it
    for phrase in s_list:
        en_sum = 1
        es_sum = 1
        simple_sum = 1
        phrase_list = phrase.split(" ")
        for i in range(0, len(phrase_list) - 2):
            pair = phrase_list[i + 1] + " " + phrase_list[i + 2]
            triple = phrase_list[i] + " " + phrase_list[i + 1] + " " + phrase_list[i + 2]
            if phrase_list[i + 2] not in tokens_probs_en:
                tokens_probs_en[phrase_list[i + 2]] = 1 / (corpus_size_en + tokens_types_en)
            if pair not in pairs_probs_en.keys():
                pairs_probs_en[pair] = 1 / pairs_types_en
                pairs_probs_en[pair] *= tokens_probs_en[phrase_list[i + 2]]
            if triple not in triples_probs_en.keys():
                triples_probs_en[triple] = 1 / triples_types_en
                triples_probs_en[triple] = (delta_a * triples_probs_en[triple]) + \
                                           (delta_b * pairs_probs_en[pair]) + (
                                                   delta_c * tokens_probs_en[phrase_list[i + 2]])
            en_sum *= triples_probs_en[triple]

            if phrase_list[i + 2] not in tokens_probs_es:
                tokens_probs_es[phrase_list[i + 2]] = 1 / (corpus_size_es + tokens_types_es)
            if pair not in pairs_probs_es.keys():
                pairs_probs_es[pair] = 1 / pairs_types_es
                pairs_probs_es[pair] *= tokens_probs_es[phrase_list[i + 2]]
            if triple not in triples_probs_es.keys():
                triples_probs_es[triple] = 1 / triples_types_es
                triples_probs_es[triple] = (delta_a * triples_probs_es[triple]) + \
                                           (delta_b * pairs_probs_es[pair]) + (
                                                   delta_c * tokens_probs_es[phrase_list[i + 2]])
            es_sum *= triples_probs_es[triple]

            if phrase_list[i + 2] not in tokens_probs_simple:
                tokens_probs_simple[phrase_list[i + 2]] = 1 / (corpus_size_simple + tokens_types_simple)
            if pair not in pairs_probs_simple.keys():
                pairs_probs_simple[pair] = 1 / pairs_types_simple
                pairs_probs_simple[pair] *= tokens_probs_simple[phrase_list[i + 2]]
            if triple not in triples_probs_simple.keys():
                triples_probs_simple[triple] = 1 / triples_types_simple
                triples_probs_simple[triple] = (delta_a * triples_probs_simple[triple]) + \
                                               (delta_b * pairs_probs_simple[pair]) + (
                                                       delta_c * tokens_probs_simple[phrase_list[i + 2]])
            simple_sum *= triples_probs_simple[triple]
        max_prob = max(en_sum, es_sum, simple_sum)
        if max_prob == en_sum:
            print("'" + phrase + "' - english corpus")
        elif max_prob == es_sum:
            print("'" + phrase + "' - espanol corpus")
        else:
            print("'" + phrase + "' - simple english corpus")


def prepare_tri_probs(corpora):
    # Count each straight triple in the corpus; key = triple, value = counter
    triples_counter = {}
    all_tokens = corpora[0][1].split(" ")
    # -3 because last item doesn't have a triple
    for i in range(0, len(all_tokens) - 3):
        triple = all_tokens[i] + " " + all_tokens[i + 1] + " " + all_tokens[i + 2]
        if triple not in triples_counter.keys():
            triples_counter[triple] = 1
        else:
            triples_counter[triple] += 1

    # Calculate probs for every triple; key = triple, value = probability
    pairs_counter = corpora[2][1]
    triples_probs = {}
    for key in triples_counter:
        first_pair = " ".join(key.split(" ")[0:2])
        triples_probs[key] = triples_counter[key] / pairs_counter[first_pair]

    # Calculate probs for every triple using backoff linear interpolation
    delta_a = 0.6
    delta_b = 0.2
    delta_c = 0.2
    uni_probs = corpora[1][1]
    bi_probs = corpora[2][1]
    triples_probs_backoff = {}
    for key in triples_counter:
        key_splited = key.split(" ")
        triples_probs_backoff[key] = (delta_a * triples_probs[key]) + \
                                     (delta_b * bi_probs[key_splited[1] + " " + key_splited[2]]) + (
                                             delta_c * uni_probs[key_splited[2]])
    return [triples_counter, triples_probs_backoff]


def bigrams_phrases(corpora_with_probs):
    # Sample 2,3,4 and 5 words phrases
    s_list = [random_phrase(corpora_with_probs, 2), "no fear", random_phrase(corpora_with_probs, 3),
              "a rotten apple",
              random_phrase(corpora_with_probs, 4),
              "del loves eating chocolate", random_phrase(corpora_with_probs, 5), "chipped a bone in his shoulder"]

    # Get each corpus' unigram and bigrams probs table, pair types, token types and size
    tokens_probs_en = corpora_with_probs['en'][1][1]
    pairs_probs_en = corpora_with_probs['en'][2][1]
    corpus_size_en = len(corpora_with_probs['en'][0][0].split(" "))
    tokens_types_en = len(corpora_with_probs['en'][1][0].keys())
    pairs_types_en = len(corpora_with_probs['en'][2][0].keys())

    tokens_probs_es = corpora_with_probs['es'][1][1]
    pairs_probs_es = corpora_with_probs['es'][2][1]
    corpus_size_es = len(corpora_with_probs['es'][0][0].split(" "))
    tokens_types_es = len(corpora_with_probs['es'][1][0].keys())
    pairs_types_es = len(corpora_with_probs['es'][2][0].keys())

    tokens_probs_simple = corpora_with_probs['simple'][1][1]
    pairs_probs_simple = corpora_with_probs['simple'][2][1]
    corpus_size_simple = len(corpora_with_probs['simple'][0][0].split(" "))
    tokens_types_simple = len(corpora_with_probs['simple'][1][0].keys())
    pairs_types_simple = len(corpora_with_probs['simple'][2][0].keys())

    # Decide what corpus each phrase is from
    # If "w_n-1" is not found in corpus- smooth it using (C(w_n-1) + 1) / N + V formula where C(w_n-1) = 0
    # If "w_n-1w_n" is not found in corpus- smooth it using (C(w_n-1w_n) + 1) / (C(w_n-1) + V) where C(w_n-1w_n) = 0 and C(w_n) = 0
    for phrase in s_list:
        en_sum = 1
        es_sum = 1
        simple_sum = 1
        phrase_list = phrase.split(" ")
        for i in range(0, len(phrase_list) - 1):
            pair = phrase_list[i] + " " + phrase_list[i + 1]
            if phrase_list[i] not in tokens_probs_en:
                tokens_probs_en[phrase_list[i]] = 1 / (corpus_size_en + tokens_types_en)
            if pair not in pairs_probs_en.keys():
                pairs_probs_en[pair] = 1 / pairs_types_en
                # Using chain rule
                pairs_probs_en[pair] *= tokens_probs_en[phrase_list[i]]
            en_sum *= pairs_probs_en[pair]

            if phrase_list[i] not in tokens_probs_es:
                tokens_probs_es[phrase_list[i]] = 1 / (corpus_size_es + tokens_types_es)
            if pair not in pairs_probs_es.keys():
                pairs_probs_es[pair] = 1 / pairs_types_es
                # Using chain rule
                pairs_probs_es[pair] *= tokens_probs_es[phrase_list[i]]
            es_sum *= pairs_probs_es[pair]

            if phrase_list[i] not in tokens_probs_simple:
                tokens_probs_simple[phrase_list[i]] = 1 / (corpus_size_simple + tokens_types_simple)
            if pair not in pairs_probs_simple.keys():
                pairs_probs_simple[pair] = 1 / pairs_types_simple
                # Using chain rule
                pairs_probs_simple[pair] *= tokens_probs_simple[phrase_list[i]]
            simple_sum *= pairs_probs_simple[pair]
        max_prob = max(en_sum, es_sum, simple_sum)
        if max_prob == en_sum:
            print("'" + phrase + "' - english corpus")
        elif max_prob == es_sum:
            print("'" + phrase + "' - espanol corpus")
        else:
            print("'" + phrase + "' - simple english corpus")


def prepare_bi_probs(corpora):
    # Count each straight pair in the corpus; key = pair, value = counter
    # Keep in "types" the number of token types
    pairs_counter = {}
    types = len(list(corpora[1][0].keys()))

    all_tokens = corpora[0][1].split(" ")
    # -2 because last item doesn't have a pair
    for i in range(0, len(all_tokens) - 2):
        pair = all_tokens[i] + " " + all_tokens[i + 1]
        if pair not in pairs_counter.keys():
            pairs_counter[pair] = 1
        else:
            pairs_counter[pair] += 1

    tokens_counter = corpora[1][0]
    tokens_probs = corpora[1][1]

    # Calculate probs for every pair using Laplace Smoothing method; key = pair, value = probability
    pairs_probs = {}
    for key in pairs_counter:
        first_token = key.split(" ")[0]
        first_token_prob = tokens_probs[first_token]

        # Smoothing pair probability formula - P*(w_n-1w_n) = ((C(w_n-1w_n) + 1) * C(w_n-1)) / (C(w_n-1) + V)
        pairs_probs[key] = (pairs_counter[key] + 1) / (tokens_counter[first_token] + types)

        # Using chain rule
        pairs_probs[key] *= first_token_prob

    return [pairs_counter, pairs_probs]


def unigrams_phrases(corpora_with_probs):
    # Sample 1,2,3 and 5 words phrases
    s_list = [random_phrase(corpora_with_probs, 1), "romance", random_phrase(corpora_with_probs, 2), "football field",
              random_phrase(corpora_with_probs, 3),
              "cup of tea", random_phrase(corpora_with_probs, 5), "it was made in china"]

    # Get each corpus unigram probs table
    en_probs = corpora_with_probs['en'][1][1]
    es_probs = corpora_with_probs['es'][1][1]
    simple_probs = corpora_with_probs['simple'][1][1]

    # Decide what corpus each phrase is from
    # When a token is not found on the corpus- its prob will be 0
    for phrase in s_list:
        en_sum = 1
        es_sum = 1
        simple_sum = 1
        for word in phrase.split(" "):
            if not word in en_probs.keys():
                en_sum *= 0
            else:
                en_sum *= en_probs[word]
            if not word in es_probs.keys():
                es_sum *= 0
            else:
                es_sum *= es_probs[word]
            if not word in simple_probs.keys():
                simple_sum *= 0
            else:
                simple_sum *= simple_probs[word]
        max_prob = max(en_sum, es_sum, simple_sum)
        if max_prob == en_sum:
            print("'" + phrase + "' - english corpus")
        elif max_prob == es_sum:
            print("'" + phrase + "' - espanol corpus")
        else:
            print("'" + phrase + "' - simple english corpus")


def prepare_uni_probs(key_value):
    # Count each token in the corpus; key = token, value = counter
    tokens_counter = {}
    all_tokens = key_value[1].split(" ")
    for token in all_tokens:
        if token not in tokens_counter.keys():
            tokens_counter[token] = 1
        else:
            tokens_counter[token] += 1

    # Calculate each token's probability in the corpus; key = token, value = probability
    tokens_probs = {}
    for key in tokens_counter:
        # Calculate using the formula P(w_n) = C(w_n) / N
        tokens_probs[key] = tokens_counter[key] / len(all_tokens)

    return [tokens_counter, tokens_probs]


def preparation_stage(content):
    # We create 3 lists/strings -
    # "sentences_len" - length of each sentence
    # "sentences_len_counter" - counts for each length got in "content" how many sentences there are in this length
    # "new_content" - same as content just with added <s> at the beginning of the a sentence and </s> at the end
    sentences_len = {}
    sentences_len_counter = {}
    new_content = ""
    for line_number, line_content in enumerate(content.splitlines()):
        line_length = len(line_content.split(" "))
        sentences_len[line_number] = line_length
        if line_length not in sentences_len_counter.keys():
            sentences_len_counter[line_length] = 1
        else:
            sentences_len_counter[line_length] += 1
        new_content += "<s> " + line_content + " </s>\n"
    res = [content.replace("\n", " "), new_content.replace("\n", " "), sentences_len, sentences_len_counter]
    return res


def main():
    input_dir = sys.argv[1]
    corpora = {}
    # We'll go through all files in the input directory and add each text file content to
    # "finished corpora" in its key (which is the language the file is written in)
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), encoding='utf-8') as txtfile:
            lang = os.path.basename(txtfile.name).split("_")[0]
            if lang in corpora.keys():
                corpora[lang] += txtfile.read()
            else:
                corpora[lang] = txtfile.read()
    finished_corpora = {}
    for lang in corpora.keys():
        org_content = preparation_stage(corpora[lang])

        # Get corpus tokens list (as keys) and counting each one of them in the first returned list
        # along with their probs as the second returned list
        tokens_with_uni_probs = prepare_uni_probs(org_content)

        finished_corpora[lang] = [org_content, tokens_with_uni_probs]

        # Get corpus pairs list (as keys) and counting each one of them in the first returned list
        # along with their probs as the second returned list
        tokens_with_bi_probs = prepare_bi_probs(finished_corpora[lang])

        finished_corpora[lang].append(tokens_with_bi_probs)

        # Get corpus triples list (as keys) and counting each one of them in the first returned list
        # along with their probs as the second returned list
        tokens_with_tri_probs = prepare_tri_probs(finished_corpora[lang])

        finished_corpora[lang].append(tokens_with_tri_probs)

    # To see the random phrases recognition untag the tagged lines below

    # print("\n\n\n")
    # print("########## Phrases recognition unigrams ##########")
    # unigrams_phrases(finished_corpora)
    # print("\n\n\n")
    # print("########## Phrases recognition bigrams ##########")
    # bigrams_phrases(finished_corpora)
    # print("\n\n\n")
    # print("########## Phrases recognition trigrams ##########")
    # trigrams_phrases(finished_corpora)

    random_sentences(finished_corpora, 1)
    source_file.writelines("\n")
    random_sentences(finished_corpora, 2)
    source_file.writelines("\n")
    random_sentences(finished_corpora, 3)
    source_file.writelines("\n")

    # Create one corpus of all three corpuses
    united_corpus = ""
    for lang in corpora:
        united_corpus += corpora[lang]
    org_content = preparation_stage(united_corpus)
    tokens_with_uni_probs = prepare_uni_probs(org_content)

    finished_corpora[lang] = [org_content, tokens_with_uni_probs]

    tokens_with_bi_probs = prepare_bi_probs(finished_corpora[lang])

    finished_corpora[lang].append(tokens_with_bi_probs)

    tokens_with_tri_probs = prepare_tri_probs(finished_corpora[lang])

    source_file.writelines("\n")
    source_file.writelines("Bigrams model based on complete dataset (English, Spanish, Simple English):" + "\n\n")
    for i in range(0, 5):
        random_length = random.choices(list(org_content[3].keys()), list(org_content[3].values()), k=1)
        sentence = random_bigrams_sentences(tokens_with_bi_probs[1], random_length)
        source_file.writelines(sentence)
        source_file.writelines("\n")

    source_file.writelines("\n")
    source_file.writelines("Trigrams model based on complete dataset (English, Spanish, Simple English):" + "\n\n")
    for i in range(0, 5):
        random_length = random.choices(list(org_content[3].keys()), list(org_content[3].values()), k=1)
        sentence = random_trigrams_sentences(tokens_with_tri_probs[1], random_length)
        source_file.writelines(sentence)
        source_file.writelines("\n")

    source_file.close()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("The run has taken %s seconds" % math.floor((time.time() - start_time)))
