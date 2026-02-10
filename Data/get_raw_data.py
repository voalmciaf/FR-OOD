import pandas as pd
import json
import math
import re


CHARACTERS = [chr(i) for i in range(65, 91)]
TRICKY_VARIABLE = ["NO", "OT", "AN", "ND"]
NEW_ABDUCTION_DESCRIPTION = "Instruction: For each goal, identify which premises directly lead to the goal. Then, trace back what the true value of the atoms must be to make each of the goal true. Only the atoms in the 'known atoms' are known but their values are not shown. Finally, return the reachable goals with the true values of the known atoms that make it true. Please enclose the final answer with <answer><answer>. All the intermediate thinking steps should be enclosed in <think><think> tags."
NEW_DEDUCTION_DESCRIPTION = "Please list the truth value of each variable to make the whole conjunction true using a JSON dictionary, which maps variable names to their truth values, then enclose the answer in <answer><answer>. Please put all the intermediate reasoning steps in <think><think>."
NEW_INDUCTION_DESCRIPTION = "Please enclose the answer in <answer><answer>, and put all the intermediate reasoning steps in <think><think>."


data_list_train = []
data_list_dev = []
data_list_test = []
for did in range(1, 5):
    df = pd.read_parquet(f'Meta-Ability-Alignment/Training/dataset/Deduction/{did}/train.parquet')
    
    data_len = len(df)
    tenth = math.ceil(data_len / 10)
    for i in range(len(df)):
        data_id = f"{did}_{i}"

        context = df['puzzle_text'][i]
        segments = context.split("-")
        new_description = "Below are some formulas connected by conjunctions:\n" + " ∧ ".join(segments[1:])
        new_description = new_description.replace("Please list the truth value of each variable\n", NEW_DEDUCTION_DESCRIPTION)
        new_answer = df['solution_text'][i]
        pattern = r'\s+([A-Z]+)\s+is\s+(True|False)'
        matches = re.findall(pattern, new_answer)
        # print(matches)

        str_list = []
        for v, b in matches:
            str_list.append(f'"{v}": "{b}"')
        
        new_answer = "{\n" + ",\n".join(str_list) + "\n}"

        if i < tenth:
            data_list_test.append({
                'id': data_id,
                'context': new_description,
                'question': NEW_DEDUCTION_DESCRIPTION,
                'answer': new_answer,
            })
        elif i < 2 * tenth:
            data_list_dev.append({
                'id': data_id,
                'context': new_description,
                'question': NEW_DEDUCTION_DESCRIPTION,
                'answer': new_answer,
            })
        else:
            data_list_train.append({
                'id': data_id,
                'context': new_description,
                'question': NEW_DEDUCTION_DESCRIPTION,
                'answer': new_answer,
            })


with open('raw_data/deductive_data_train.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_train, f, ensure_ascii=False, indent=4)

with open('raw_data/deductive_data_dev.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_dev, f, ensure_ascii=False, indent=4)

with open('raw_data/deductive_data_test.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_test, f, ensure_ascii=False, indent=4)



data_list_test = []
data_list_dev = []
data_list_train = []
for did in range(1, 6):
    df = pd.read_parquet(f'Meta-Ability-Alignment/Training/dataset/Abduction/{did}/train.parquet')

    context = df['puzzle_text'][i]
    premises = context.split("\nknown_atoms")[0][10:]
    # print(premises)


    atoms = context.split("\nknown_atoms: ")[1].split("\n")[0]
    

    matched_atoms = re.findall(r"'(.*?)'", atoms)

    goals = context.split("\ngoals: ")[1].split("\n")[0]
    matched_goals = re.findall(r"'(.*?)'", goals)

    matched_atoms_and_goal = matched_atoms + matched_goals

    if len(matched_atoms_and_goal) >= 26:
        print("Too many atoms!!!!!")

    # print(matched_atoms)
    replacement = {}
    new_atoms = []
    for c in matched_atoms:
        if len(c) > 1:
            replace_characters = [value for _, value in replacement.items()]
            for i in range(len(CHARACTERS)):
                if CHARACTERS[i] not in replace_characters and CHARACTERS[i] not in matched_atoms_and_goal:
                    replacement[c] = CHARACTERS[i]
                    new_atoms.append(CHARACTERS[i])
                    break
        else:
            new_atoms.append(c)
    # new_atoms = str(new_atoms)
    # print(new_atoms)
    
    

    # print(new_premises)
    new_goals = []
    for g in matched_goals:
        if len(g) > 1:
            replace_characters = [value for _, value in replacement.items()]
            for i in range(len(CHARACTERS)):
                if CHARACTERS[i] not in replace_characters and CHARACTERS[i] not in matched_atoms_and_goal:
                    replacement[g] = CHARACTERS[i]
                    new_goals.append(CHARACTERS[i])
                    break
        else:
            new_goals.append(g)

    new_premises = premises
    for c in replacement:
        if c in TRICKY_VARIABLE:
            new_premises = new_premises.replace("AND", "&")
            new_premises = new_premises.replace("NOT", "~")
            new_premises = new_premises.replace(c, replacement[c])
            new_premises = new_premises.replace("~", "NOT")
            new_premises = new_premises.replace("&", "AND")
        else:
            new_premises = new_premises.replace(c, replacement[c])

    new_context = f"Premises: {new_premises}\nKnown Atoms: {new_atoms}\nGoals: {new_goals}\n\n{NEW_ABDUCTION_DESCRIPTION}"

    new_answer = df['solution_text'][i].replace("is reachable", "+").replace("is unreachable", "-")  
    for c in replacement:
        new_answer = new_answer.replace(c, replacement[c])
    new_answer = new_answer.replace("+", "is reachable").replace("-", "is unreachable")
    
    data_len = len(df)
    tenth = math.ceil(data_len / 10)
    for i in range(len(df)):
        difficulty = df['difficulty'][i]
        problem_id = df['problem_id'][i]
        data_id = f"{difficulty}_{problem_id}_{i}"

        if i < tenth:
            data_list_test.append({
                'id': data_id,
                'context': new_context,
                'question': NEW_ABDUCTION_DESCRIPTION,
                'answer': new_answer,
            })
        elif i < 2 * tenth:
            data_list_dev.append({
                'id': data_id,
                'context': new_context,
                'question': NEW_ABDUCTION_DESCRIPTION,
                'answer': new_answer,
            })
        else:
            data_list_train.append({
                'id': data_id,
                'context': new_context,
                'question': NEW_ABDUCTION_DESCRIPTION,
                'answer': new_answer,
            })


with open('raw_data/abductive_data_train.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_train, f, ensure_ascii=False, indent=4)

with open('raw_data/abductive_data_dev.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_dev, f, ensure_ascii=False, indent=4)

with open('raw_data/abductive_data_test.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_test, f, ensure_ascii=False, indent=4)


data_list_test = []
data_list_dev = []
data_list_train = []
for did in range(1, 6):
    df = pd.read_parquet(f'Meta-Ability-Alignment/Training/dataset/Induction/{did}/train.parquet')
    
    data_len = len(df)
    tenth = math.ceil(data_len / 10)
    for i in range(len(df)):
        difficulty = df['difficulty'][i]
        problem_id = df['problem_id'][i]
        data_id = f"{difficulty}_{problem_id}_{i}"

        if i < tenth:
            data_list_test.append({
                'id': data_id,
                'context': df['puzzle_text'][i] + " " + NEW_INDUCTION_DESCRIPTION,
                'question': "Based on the sequence above, what is the value at the question mark? " + NEW_INDUCTION_DESCRIPTION,
                'answer': str(df['solution_text'][i]),
            })
        elif i < 2 * tenth:
            data_list_dev.append({
                'id': data_id,
                'context': df['puzzle_text'][i] + " " + NEW_INDUCTION_DESCRIPTION,
                'question': "Based on the sequence above, what is the value at the question mark? " + NEW_INDUCTION_DESCRIPTION,
                'answer': str(df['solution_text'][i]),
            })
        else:
            data_list_train.append({
                'id': data_id,
                'context': df['puzzle_text'][i] + " " + NEW_INDUCTION_DESCRIPTION,
                'question': "Based on the sequence above, what is the value at the question mark? " + NEW_INDUCTION_DESCRIPTION,
                'answer': str(df['solution_text'][i]),
            })


with open('raw_data/inductive_data_train.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_train, f, ensure_ascii=False, indent=4)

with open('raw_data/inductive_data_dev.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_dev, f, ensure_ascii=False, indent=4)

with open('raw_data/inductive_data_test.json', 'w', encoding='utf-8') as f:
    json.dump(data_list_test, f, ensure_ascii=False, indent=4)

