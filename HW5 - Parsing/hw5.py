import os
import sys

used_non_terminals = {}

original_non_terms = {}

first_symbol = None


# This class represents the parse tree created during the CYK Algorithm
class Node:
    def __init__(self, left, right, name, non_terminal=None):
        self.left = left
        self.right = right
        self.name = name
        self.non_terminal = non_terminal


def cyk_algo(rules, sentence, output_file):
    n = len(sentence.split())
    sentence = sentence.split()

    # Initialize backpointers, nodes and probabilities tables
    T = [["" for j in range(n)] for i in range(n)]
    N = [["" for j in range(n)] for i in range(n)]
    Probs = [[0 for j in range(n)] for i in range(n)]

    # Initialize main diagonal of every table with the terminals values
    for j in range(0, n):

        # Go over the rules for the j iteration
        for left_side, rule in rules.items():
            for right_side in rule:

                # If a terminal is found, keep its name on the backpointers table,
                # its node representation on the nodes table and probability on the probabilities table
                if len(right_side) == 1 and right_side[0] == sentence[j]:
                    T[j][j] = left_side
                    N[j][j] = Node(None, None, right_side[0], left_side)
                    Probs[j][j] = 1 / len(rule)

    # Filling in the table
    for j in range(0, n):

        for i in range(j, -1, -1):
            # Iterate over the range i to j + 1
            # k = i
            for k in range(i, j + 1):

                # Go over the rules for the k iteration
                for lhs, rule in rules.items():
                    for rhs in rule:

                        # If a terminal is found and its deriving rule probability, multiplied by the probabilities
                        # of the derived terminals/non-terminals, is higher than the current probability
                        # in Probs[i][j], replace Probs[i][j] with curr_prob, put in on the current cell of
                        # backpointers and its node representation on the nodes table
                        curr_prob = (1 / len(rule)) * Probs[i][k - 1] * Probs[k][j]
                        if i != 0 and j != n - 1 and lhs == first_symbol:
                            continue
                        if len(rhs) == 2 and rhs[0] in T[i][k - 1] and rhs[1] in T[k][j] and Probs[i][k - 1] > 0 and \
                                Probs[k][j] > 0 and curr_prob > Probs[i][j]:
                            Probs[i][j] = curr_prob
                            T[i][j] = lhs
                            N[i][j] = Node(N[i][k - 1], N[k][j], lhs)

    # If left side of first rule appears only one time-
    # string is valid only if the 0,n-1 cell equals to left side of first rule
    sentence = ' '.join(sentence)
    if first_symbol and T[0][n - 1] == first_symbol:
        output_tree = print_tree(N[0][n - 1])
        output_file.writelines(str(sentence) + "\n" + output_tree + "\nParse's tree probability: " + str(Probs[0][n-1]))
        output_file.writelines("\n\n")
        return

    if len(T[0][n - 1]) != 0:
        output_tree = print_tree(N[0][n - 1])
        output_file.writelines(str(sentence) + "\n" + output_tree + "\nParse's tree probability: " + str(Probs[0][n-1]))
        output_file.writelines("\n\n")
    else:
        output_file.writelines(sentence + "\n")
        output_file.writelines("not found")
        output_file.writelines("\n\n")


def print_tree(node):
    # Check for a terminal
    if node.left is None and node.right is None:
        if node.non_terminal in original_non_terms.keys():
            return "(" + node.non_terminal + " " + node.name + ")"
        else:
            return " " + node.name + ""

    # count check whether a non_terminal comes from two different non terminals
    # if it does- don't write the non_terminal
    count = 0
    if node.name not in original_non_terms.keys():
        for key in original_non_terms.keys():
            if node.name in original_non_terms[key]:
                count += 1
                node.name = key
    if count > 1:
        return " " + print_tree(node.left) + " " + print_tree(node.right) + " "
    else:
        return "(" + node.name + " " + print_tree(node.left) + " " + print_tree(node.right) + ")"


def list_to_dict(rules):
    dict_rules = {}
    for rule in rules:
        splited_rule = rule.split(" -> ")
        if not splited_rule[0] in dict_rules.keys():
            dict_rules[splited_rule[0]] = []
        dict_rules[splited_rule[0]].append(splited_rule[1])
    return dict_rules


def terminal_with_non_terminal(rules):
    new_rules = []
    for rule in rules:
        if not rule or rule.isspace():
            continue
        splited_rule = rule.replace("\n", "").split(" -> ")
        left_side = splited_rule[0]
        right_side = splited_rule[1]
        # Cases of "a S b", means terminals and non-terminals in right side
        if any(x.isupper() for x in right_side) and any(x.islower() for x in right_side):
            new_right = []
            for elem in right_side.split():
                # In case we have a terminal before or after non-terminal
                # add new non-terminal Xi and create a rule where Xi on the left side
                # and a terminal on the right side. Xi will replace the terminal
                # at its original place
                if elem[0].islower():
                    new_non_terminal = "X" + str(1)
                    while new_non_terminal in used_non_terminals.values():
                        new_non_terminal += str(1)
                    used_non_terminals[elem] = new_non_terminal
                    new_right.append(new_non_terminal)
                    new_rules.append(new_non_terminal + " -> " + elem)
                    if not left_side in original_non_terms.keys():
                        original_non_terms[left_side] = []
                    original_non_terms[left_side].append(elem)
                else:
                    new_right.append(elem)
            new_rules.append(left_side + " -> " + " ".join(new_right))
        # Cases of "a b", means only terminals in right side
        elif right_side.islower() and len(right_side.split()) > 1:
            new_right = []
            for elem in right_side.split():
                if elem in used_non_terminals.keys():
                    new_right.append(used_non_terminals[elem])
                else:
                    new_non_terminal = "X" + str(1)
                    while new_non_terminal in used_non_terminals.values():
                        new_non_terminal += str(1)
                    used_non_terminals[elem] = new_non_terminal
                    new_right.append(new_non_terminal)
            new_rules.append(left_side + " -> " + " ".join(new_right))
        else:
            new_rules.append(rule)
    return new_rules


# Check whether a given rule is unit rule
def is_unit(rule):
    splited_rule = rule.split()
    if len(splited_rule) > 3:
        return False
    return splited_rule[2][0].isupper()


def eliminate_one_unit_rule(rules):
    new_rules = []
    loop = False
    for rule in rules:
        # In case we're in an empty line
        if not rule or rule.isspace():
            continue
        # If the rule is not unit production, add him to 'new_rules'
        # else we'll have to check the rules in one more loop so loop = True
        # and continue to the next iteration (except S' which is a starter rule)
        if rule.startswith("S'") or not is_unit(rule):
            new_rules.append(rule)
            continue
        loop = True

        # At this point we have a unit rule, it's not added to 'new_rules'
        splited_rule = rule.replace("\n", "").split(" -> ")
        left_side = splited_rule[0]
        right_side = splited_rule[1]

        # Find the rules where the right side non-terminal in this rule
        # is the left side and add them to 'new_rules'
        all_rules_of_right_side = [i for i in rules if right_side is i.split(" -> ")[0]]
        for right_right in all_rules_of_right_side:
            if not right_right.startswith("S'"):
                new_rules.append(left_side + " -> " + right_right.split(" -> ")[1])
    return new_rules, loop


def eliminate_one_unit_rule_loop(rules):
    keep_loop = True
    # Loop over the rules until there are no more one unit rules
    while keep_loop:
        (rules, keep_loop) = eliminate_one_unit_rule(rules)
    return rules


def multiple_non_terminals_on_right_side(rules):
    new_rules = {}
    for rule in rules:
        # In case we're in an empty line
        if not rule or rule.isspace():
            continue
        splited_rule = rule.replace("\n", "").split(" -> ")
        left_side = splited_rule[0]
        right_side = splited_rule[1]
        right_side_elements = right_side.split()
        to_continue = False

        # Go over all the rules and check the right side of each rule contains a terminal
        # If at least one is a terminal- add it to 'new_rules' and continue to next iteration
        for elem in right_side_elements:
            if elem.islower():
                if not left_side in new_rules.keys():
                    new_rules[left_side] = []
                new_rules[left_side].append(right_side)
                to_continue = True
                break
        if to_continue:
            continue
        # At this point we're in a rule if only non-terminals
        # Check whether we have more than 2 non-terminals in the right side
        # if it doesn't - continue to next iteration
        if len(right_side_elements) > 2:

            # Create a new non-terminal (only if it doesn't exist yet)
            # Loop over all the non-terminals in the right side
            # For each non-terminal (starting from the second one) create a new rule
            # where the right side is the new non-terminal and right side the rest of thr right side
            for i in range(0, len(right_side_elements)):
                new_nonterminal = "NewNon" + str(i)
                if new_nonterminal in new_rules.keys():
                    if right_side_elements[i] + " " + right_side_elements[i + 1] == new_rules[new_nonterminal]:
                        new_nonterminal += str(i)
                if i == len(right_side_elements) - 2:
                    if not left_side in new_rules.keys():
                        new_rules[left_side] = []
                    if not right_side_elements[i] + " " + right_side_elements[i + 1] in new_rules[left_side]:
                        new_rules[left_side].append(right_side_elements[i] + " " + right_side_elements[i + 1])
                    break
                else:
                    if not left_side in new_rules.keys():
                        new_rules[left_side] = []
                    if not right_side_elements[i] + " " + new_nonterminal in new_rules[left_side]:
                        new_rules[left_side].append(right_side_elements[i] + " " + new_nonterminal)
                if not left_side in original_non_terms.keys():
                    original_non_terms[left_side] = []
                original_non_terms[left_side].append(new_nonterminal)
                left_side = new_nonterminal
        else:
            if not left_side in new_rules.keys():
                new_rules[left_side] = []
            new_rules[left_side].append(right_side)

    final_rules = []
    for key in new_rules.keys():
        for value in new_rules[key]:
            final_rules.append(key + " -> " + value)
    return final_rules


def main():
    # Convert a CFG to CNF - first part of the assignment
    grammer_in = os.path.join(sys.argv[1])
    with open(grammer_in, encoding='utf-8') as txtfile:
        rules = txtfile.readlines()
    save_grammer_dir_path = os.path.join(sys.argv[2])
    grammer_out = open(save_grammer_dir_path, "w+", encoding="utf-8")

    # Keep a dict of all original rules, for later
    # convert the algorithm added rules back to their
    # original form
    for rule in rules:
        splited_rule = rule.split(" -> ")
        left_side = splited_rule[0]
        right_side = splited_rule[1]
        if left_side not in original_non_terms.keys():
            original_non_terms[left_side] = []
        original_non_terms[left_side].append(right_side.replace("\n", ""))

    # Check if starting terminal occurs on some right side
    # If it does, add "S' -> 'starting terminal'" rule as a starting rule
    global first_symbol
    first_symbol = rules[0].split(" -> ")[0]
    s_symbol = first_symbol
    for rule in rules:
        if s_symbol in rule.split():
            rules.insert(0, "S' -> " + s_symbol)
            break

    rules = eliminate_one_unit_rule_loop(rules)
    rules = multiple_non_terminals_on_right_side(rules)
    rules = terminal_with_non_terminal(rules)
    new_rules = list_to_dict(multiple_non_terminals_on_right_side(rules))

    if "S'" in new_rules.keys():
        grammer_out.writelines("S'-> " + "".join(new_rules["S'"]))
        grammer_out.writelines("\n")
        new_rules.pop("S'", None)

    new_rules = filter_unused_rules(new_rules)

    for non_terminal in new_rules.keys():
        for right in new_rules[non_terminal]:
            grammer_out.writelines(non_terminal + " -> " + right)
            grammer_out.writelines("\n")
    grammer_out.close()

    # Get all input sentences to parse
    input_sentences = os.path.join(sys.argv[3])
    with open(input_sentences, encoding='utf-8') as txtfile2:
        sentences = txtfile2.readlines()
    output_file = open(os.path.join(sys.argv[4]), "w+", encoding="utf-8")

    # Organize the dict to fit the CYK
    for key in new_rules.keys():
        l = []
        for item in new_rules[key]:
            s_item = item.split()
            if len(s_item) == 2:
                # if the first symbol appears only on the first rule, 'first_symbol' = None
                if first_symbol in s_item:
                    first_symbol = None
                item = [s_item[0], s_item[1]]
                l.append(item)
            else:
                l.append([item])
        new_rules[key] = l

    # CYK Algorithm
    for sentence in sentences:
        cyk_algo(new_rules, sentence, output_file)


def filter_unused_rules(rules):
    filtered_rules = {}
    for key in rules.keys():
        # Don't filter first non terminal
        if key == first_symbol:
            filtered_rules[key] = rules[key]
            continue
        found = False
        for value in rules.values():
            for item in value:
                if key in item.split():
                    found = True
        if found:
            filtered_rules[key] = rules[key]
    return filtered_rules


if __name__ == "__main__":
    main()
