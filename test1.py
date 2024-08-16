from utils.lean_math_utils import *
from utils.lean_theorem_utils import *

def count_lines(string):
    # Split the string into lines
    lines = string.splitlines()
    # Count the number of lines
    return len(lines)

def extract_first_case(state_pp):
    state_pp = state_pp.strip()
    if not state_pp.startswith('case'):
        return state_pp
    lines = state_pp.split('\n')
    first_case = []
    for line in lines[1:]:
        if line.strip().startswith('case'):
            break
        if line.strip() != '':
            first_case.append(line)
    return '\n'.join(first_case)


# Params:
#   hyp: tuple(name, type)
#   tactics: list(tactic)
def is_hypothesis_useful(hyp, tactics):
    for tactic in tactics:
        tokens = tokenize_lean_tactic(tactic)
        if hyp[0] in tokens:
            idx = tokens.index(hyp[0])
            if idx > 0:
                if hyp[0].startswith('h'):
                    return True
                if tokens[idx - 1] == 'exact':
                    return True
                if tokens[idx - 1] == 'at':
                    return True
                if tokens[idx - 1] == '[':
                    return True
                if idx < len(tokens) - 1 and tokens[idx + 1] == ']':
                    return True
                for operator in TargetNode.operators:
                    if operator in hyp[1]:
                        return True
    return False

def create_hypothesis_predict_data(raw_state_pp, tactics, theorem_name):
    is_case = raw_state_pp.strip().startswith('case')
    state_pp = extract_first_case(raw_state_pp)
    if is_case and count_lines(state_pp) < count_lines(raw_state_pp) - 2:
        tactics = tactics[0:1]
    #
    premise = Premise()
    premise.theorem_name = theorem_name
    premise.parse_state(state_pp)
    #
    useful_hypotheses, useless_hypotheses = [], OrderedDict()
    for hyp in premise.hypotheses.items():
        useful = is_hypothesis_useful(hyp, tactics)
        if useful:
            #print("YES:", hyp)
            useful_hypotheses.append(hyp)
        else:
            #print("NO :", hyp)
            useless_hypotheses[hyp[0]] = hyp[1]
    premise.hypotheses = useless_hypotheses
    return premise, useful_hypotheses

import os
import pickle

data_folder = '/home/mcwave/code/axiomatization/datasets/mathlib4_states_w_proof/'
file_names = os.listdir(data_folder)

fout = open('/home/mcwave/code/axiomatization/datasets/mathlib4_states_w_proof_hyp_pred.pkl', 'wb')

count = 0
for file_name in file_names:
    if not file_name.endswith("pkl"):
        continue
    count += 1
    data = []
    print("Loading", file_name)
    file_path = os.path.join(data_folder, file_name)
    fin = open(file_path, 'rb')
    while True:
        try:
            pair = pickle.load(fin)
            data.append(pair) #(pair[1][0], pair[1][2][0]))
        except:
            break
    #
    fin.close()
    print(len(data), "examples loaded")
    hyp_data = []
    for pair in data:
        state_pp = pair[1][0]
        tactics = pair[1][2]
        if tactics is None or len(tactics) == 0:
            continue
        full_path = pair[0][2]
        theorem_name = pair[0][3]
        premise, useful_hypotheses = create_hypothesis_predict_data(state_pp, tactics, theorem_name)
        if len(useful_hypotheses) == 0:
            continue
        premise.full_path = full_path
        hyp_data.append((premise, useful_hypotheses))
        pickle.dump((premise, useful_hypotheses), fout)
    #
    fout.flush()
    print(len(hyp_data), "hypotheses data found")

fout.close()