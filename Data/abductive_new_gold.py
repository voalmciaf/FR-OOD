import re
import string
import janus_swi as janus
import json


# --- Define replacement rules for CLP(B) ---
REPLACEMENTS = {
        # Operators
        ' OR ': ' + ',
        ' AND ': ' * ',
        'NOT ': '~',
        
        # Implication must be converted to an inequality constraint
        # We handle this one separately below to split Antecedent/Consequent.
        '=>': '=<' 
    }



def run_prolog_query(programme: str, query: str, goal: str) -> str:
    """
    Initializes the Prolog environment, loads the constraints, and runs the 
    specific query to find variable assignments.
    """
    try:
        # Load the Prolog code into the Janus environment
        janus.query_once(f"consult('{programme}')")

        # --- 2. Define and Execute the Query ---
        # Query: O is 1 (True), then solve for B, I, L, M, N

        
        
        # Execute the query and get an iterable of solutions
        solutions = janus.query(query)

        # --- 3. Print Results ---
        solution_count = 0
        
        # Solutions are returned as Python dictionaries (variable bindings)
        out_str = f"Goal '{goal}' is reachable. Here are the possible assignments to make it true:\n"
        for sol in solutions:
            # Sort the keys for consistent output
            sorted_sol = dict(sorted(sol.items())) 
            if 'truth' in sorted_sol:
                del sorted_sol['truth']
            
            new_dict = {}
            for key, value in sorted_sol.items():
                if key == goal:
                    continue
                new_dict[key] = True if value==1 else False

            out_str += f"Solution {solution_count + 1}: {json.dumps(new_dict)}\n"
            solution_count += 1

    except Exception as e:
        print(f"An error occurred during Prolog execution:")
        print(f"Make sure the 'janus_swi' library is installed and SWI-Prolog is accessible.")
        print(f"Error details: {e}")

    return out_str if out_str else "No solutions found."


def convert_to_swish_prolog(formulas_list):
    """
    Converts a list of text-based logic formulas into a single SWISH Prolog
    program using the CLP(B) constraints library.
    
    The formulas are assumed to be connected by conjunction (AND).
    
    Args:
        formulas_list (list): A list of text formulas in the format 
                              '(ANTECEDENT) => CONSEQUENT'.
    Returns:
        str: The complete Prolog code ready for SWISH.
    """
    
    # --- 1. Identify all unique variables ---
    # We look for all uppercase letters used as variables.
    all_variables = set()
    
    # --- 3. Generate the Prolog constraints body ---
    prolog_constraints = []
    
    for formula in formulas_list:
        # Split formula by '=>'
        if '=>' not in formula:
            prolog_constraints.append(f"    % Invalid format: {formula}")
            continue

        antecedent, consequent = formula.split('=>', 1)
        
        # Clean up parentheses and whitespace
        antecedent = antecedent.strip()#.strip('()').strip()
        consequent = consequent.strip()#.strip('()').strip()
        
        # Apply standard operator replacements to the antecedent
        clpb_antecedent = antecedent
        for text_op, clpb_op in REPLACEMENTS.items():
            clpb_antecedent = clpb_antecedent.replace(text_op, clpb_op)
        variables_in_formula = re.findall(r'[A-Z]', clpb_antecedent)
        all_variables.update(variables_in_formula)

        # Construct the final constraint
        # sat(Antecedent_CLPB =< Consequent)
        constraint = f"    sat({clpb_antecedent} =< {consequent})"
        prolog_constraints.append(constraint)
    # Sort them for a clean, consistent predicate signature
    sorted_vars = sorted(list(all_variables))
    var_string = ", ".join(sorted_vars)

    if not sorted_vars:
        return f"% ERROR: No variables found in the provided formulas.\n% Input: {formulas_list}"

    constraints_body = ",\n".join(prolog_constraints)

    # --- 4. Assemble the final SWISH Prolog program ---
    prolog_code = f"""
:- use_module(library(clpb)).


logic_constraints({var_string}) :-
{constraints_body}.
"""

    return prolog_code, sorted_vars


if __name__ == "__main__":
    with open("raw_data/abductive_data_test.json", "r") as f:
        data = json.load(f)

    new_data = []
    for sample in data:
        context = sample["context"]
        gold_answer = sample["answer"]
        reach_match = re.search(r'\)\s*(.+?)\s*is reachable', gold_answer)
        if reach_match:
            goal_string = reach_match.group(1)
        # print(goal_string)



        match = re.search(r'\[([^\]]+)\]', context)
        if match:
            first_bracketed_string = match.group(1)
            formulas = re.findall(r"'([^']*)'", first_bracketed_string)
            prolog, variables = convert_to_swish_prolog(formulas)
            # print(prolog)

            query_var = [v for v in variables if v != goal_string]
            all_vars = ", ".join(variables)
            query_var_str = ", ".join(query_var)
            query = f"{goal_string}=1, logic_constraints({all_vars}), labeling([{query_var_str}])."

            with open("temp.pl", "w") as f:
                f.write(prolog)
            
            output = run_prolog_query("temp.pl", query, goal_string)
            # print(output)

            new_data.append({
                "id": sample["id"],
                "context": context,
                "reachable_goal": goal_string,
                "prolog": prolog,
                "variables": variables,
                "question": sample["question"],
                "answer": output,
                "old_answer": gold_answer
            })
    with open("raw_data/abductive_data_test_new_gold.json", "w") as f:
        json.dump(new_data, f, indent=4)