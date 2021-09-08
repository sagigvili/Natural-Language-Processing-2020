import re
import wikipedia
import sys
import os.path

save_dir_path = os.path.join(sys.argv[3])
curr_language = None


def last_fixers(content):
    # Getting rid of extra chars added to "..." and ".."
    new_content = re.sub(r'(\@\#)(\.\.\.)(\#\@)|(\@\#)(\.\.)(\#\@)', r'\2', content)

    # There were some upper-case letters coming after lower-case ones, add whitespace between them
    new_content = re.sub(r'([a-z])([A-Z])', r'\1 \2', new_content)

    # Take care of titles- every title in a line of its own
    new_content = re.sub(r'(=?=?==.*?==?=?=?=)', r'\n\1\n', new_content)

    # Remove unnecessary whitespace after some line-breaks
    new_content = re.sub(r'\n ', r'\n', new_content)

    # Handle any last one capital letter initials
    new_content = re.sub(r'( [A-Z]\.)(\n)', r'\1', new_content)

    # Handle situations where a line break is necessary
    new_content = re.sub(r'( \.)([A-Z])', r'\1\n\2', new_content)
    new_content = re.sub(r'( \.)( )([A-Z])', r'\1\n\3', new_content)

    new_content = re.sub(r'([a-zA-Z1-9])( )(\')([a-z])', r'\1\3\4', new_content)
    new_content = re.sub(r'( \.)(\")( )([A-Z])', r'\1 \2\n\4', new_content)

    # Remove any extra chars added for initials
    new_content = re.sub(r'\#\$', r'', new_content)

    # Remove any remaining empty lines
    new_content = re.sub(r'\n\n|\n\n\n|\n\n\n\n', r'\n', new_content)
    new_content = re.sub(r'\n\n|\n\n\n|\n\n\n\n', r'\n', new_content)

    # Handle whitespace, dot, number/capital-letter, whitespace
    # Most of the time a break-line is necessary
    new_content = re.sub(r'( \.)([A-Z1-9] )', r'\1\n\2', new_content)
    new_content = re.sub(r'( \. )(\" [A-Z])', r'\1\n\2', new_content)

    # Handled bad written initials
    new_content = re.sub(r'( [A-Z])( )(\.)(\n)(\w*)( \.\n)', r'\1\3\5\6', new_content)

    new_content = re.sub(r'( \')(.*)(\' )', r'\1 \2 \3', new_content)
    new_content = re.sub(r'( \( \")', r'\1 ', new_content)
    new_content = re.sub(r'( \:)([a-zA-Z])', r'\1 \2', new_content)
    new_content = re.sub(r'([a-z])(\' \,)', r'\1 \2', new_content)
    new_content = re.sub(r'(\')(\.)([A-Z])', r'\1 \2\n\3', new_content)

    # Handle hardcoded known terms with dot
    new_content = re.sub(r'Dr \.\n', r'Dr.', new_content)
    new_content = re.sub(r'Prof \.\n', r'Prof.', new_content)
    new_content = re.sub(r'Mr \.\n', r'Mr.', new_content)
    new_content = re.sub(r'Ms \.\n', r'Ms.', new_content)
    new_content = re.sub(r'Bros \.\n', r'Bros.', new_content)
    new_content = re.sub(r' no \.\n', r' no.', new_content)
    new_content = re.sub(r' vol \. ', r' vol. ', new_content)

    new_content = re.sub(r'( \. )(\")(\n)([A-Z])', r'\1\n\2 \4', new_content)
    new_content = re.sub(r'( \.)( )(\n)(\")( )([A-Z])', r'\1\4\n\6', new_content)

    new_content = re.sub(r'St \.\n', r'St.', new_content)

    return new_content[:-1]


def tokenization(sentence):
    tokenaized_sentence = []
    if sentence.endswith("."):
        sentence = sentence[:-1] + " ."
    # Take care of "..." and ".."
    sentence = re.sub(
        r'(\.\.\.|\.\.)', r'@#\1#@',
        sentence)
    # Take care of shorten names like "H. C. A. Harisson"
    sentence = re.sub(r' ([A-Z]\.) ([a-zA-Z]\.) ([a-zA-Z]\.) (\w*)', r' \1\2\3\4#$ ', sentence)

    # Take care of shorten names like "P. c. cinereus"
    sentence = re.sub(r' ([A-Z]\.) ([a-zA-Z]\.) (\w*)', r' \1\2\3#$ ', sentence)

    # Take care of shorten names like "P. cinereus"
    sentence = re.sub(r' ([A-Z]\.) (\w*)', r' \1\2#$ ', sentence)

    # Take care of initials like "e.g.", "a.k.a.", "U.S.A.", "v."
    sentence = re.sub(
        r'([a-zA-Z]\.[a-zA-Z]\.)( )', r'\1#$ ',
        sentence)
    sentence = re.sub(
        r'([a-zA-Z]\.[a-zA-Z]\.[a-zA-Z])( )', r'\1.#$ ',
        sentence)
    sentence = re.sub(
        r'([a-zA-Z]\.[a-zA-Z]\.[a-zA-Z]\.)( )', r'\1#$ ',
        sentence)
    sentence = re.sub(
        r'( )([a-z]\.)( )', r'\1\2#$ ',
        sentence)
    sentence = re.sub(
        r'(\()([a-z]\.)( )', r'\1\2#$ ',
        sentence)

    # Take care of sentences contains only "*some sentence*"
    sentence = re.sub(r'(\")(.*)(\.)(\")', r'\1 \2 \3 \4', sentence)
    sentence = re.sub(r'(\")(.*)(\")', r'\1 \2 \3', sentence)

    # Take care of punctuation
    sentence = re.sub(r'(\)|\–|\" |\"\.|\.\"|\"\,|\.\"|\"\)|\. |\?|\!|\}|\]|\, |\:|\;|\— |\' )', r' \1', sentence)
    sentence = re.sub(r'(\(|\–| \"|\(\"|\{|\[| \—| \')', r'\1 ', sentence)
    sentence = re.sub(r'([a-z1-9])(\,)(\[)', r'\1 \2 \3', sentence)

    # Take care of situations where there should be new line and not caught in punctuation before
    sentence = re.sub(r'([a-z1-9])(\.)([A-Z])', r'\1 \2 \3', sentence)
    sentence = re.sub(r'([a-z1-9])(\.)(\[)', r'\1 \2 \3', sentence)
    word_tokens = sentence.split(" ")
    for token in word_tokens:
        if re.search(r'[a-zA-Z]\#\$', token):
            tokenaized_sentence.append(re.sub(r'([a-zA-Z])(\#\$)', r'\1', token))
        else:
            tokenaized_sentence.append(token)
    return tokenaized_sentence


def make_tokens(content):
    tokens_list = []
    # Add whitespace to missing places
    content = re.sub(r'([a-z])(\.[A-Z]) ', r'\1 \2', content)
    content = re.sub(r'(\))(\.[A-Z]) ', r'\1 \2', content)
    content = re.sub(r'([A-Z])(\.[A-Z]) ', r'\1 \2', content)

    # Remove unnecessary punctuation
    content = re.sub("\(\)|\( \)", "", content)
    content.replace(",,", ",")

    sentences = content.split("\n")
    temp = ""
    for sentence in sentences:
        if not sentence:
            continue
        res = tokenization(sentence)
        res = list(filter(None, res))  # Filter any empty cells in the list
        if not res:  # Filter any empty sentences
            continue
        elif re.search(r'(=?=?==.*?==?=?=)', str(res)):
            res = " ".join(res)
        # On this stage we have some cells with a string and others with lists of strings
        # We have different approach for each occasion
        if type(res) is list:
            tokens_list.append(" ".join(res))
        else:
            tokens_list.append(res)
    return tokens_list


def order_sentences(content):
    # Remove all "displaystyle" and "frac" html tags
    new_content = re.sub(r'.*{\\displaystyle.*\n', "", content)
    new_content = re.sub(r'.*{\\frac.*\n', "", new_content)
    new_content = make_tokens(new_content)
    temp = ""
    for token in new_content:
        if token and not token.isspace():
            if len(token) == 1:
                temp += token + " "
            else:
                temp += token + "\n"
    new_content = re.sub(r'(\n)(. )', r'\2', temp)
    return last_fixers(new_content)


def make_text_file(value, lang):
    final_dir_path = save_dir_path + "\\" + lang + "_" + value.lower().replace(" ", "_") + ".txt"
    page = None
    # If the value contains only upper case letters, use search
    if value.isupper():
        for item in wikipedia.search(value):
            try:
                page = wikipedia.page(item)
                break
            except:
                pass
    else:
        # Try get page by value
        # If fails- get by search
        # If search fails- get from exception options
        # If exception options fails- return empty output page
        try:
            page = wikipedia.page(value)
        except:
            try:
                item = wikipedia.search(value)
                page = wikipedia.page(item[0])
            except wikipedia.exceptions.DisambiguationError as e:
                try:
                    page = wikipedia.page(e.options[0])
                except:
                    pass
            except wikipedia.exceptions.PageError:
                pass
            except:
                pass
    if not page:
        return
    output_file = open(final_dir_path, "w", encoding="utf-8")
    no_spaces_content = ""
    for line in page.content.split("\n"):
        if line and not line.isspace():
            no_spaces_content += line + "\n"
    output_file.writelines(order_sentences(no_spaces_content[:-1]))
    output_file.close()


def main():
    # Read values input file
    values_file = open(sys.argv[1])
    values_input = values_file.read().splitlines()
    values_file.close()

    # Read languages input file
    langs_input_file = open(sys.argv[2])
    langs_input = langs_input_file.read().splitlines()
    langs_input_file.close()

    for lang in langs_input:
        wikipedia.set_lang(lang)
        for value in values_input:
            make_text_file(value, lang)


if __name__ == "__main__":
    main()
