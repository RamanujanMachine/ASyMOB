import csv
import json
import sympy as sp
import re
import random
import itertools

# Load CSV file containing seed questions and maximal symbolic perturbations. 
csv_file_path = 'Seed_and_Max_Symbolic_Perturbations.csv'

with open(csv_file_path, 'r', encoding='utf-8') as cf:
    data = list(csv.DictReader(cf))  # Read CSV into list of dicts
cur_data_len = len(data)
print("Length of initial data: ", cur_data_len)  # Print initial dataset size

# Define symbolic variables that appear in the symbolic perturbations - and will be replaced by various expressions during variant generation.
x, A, B, F, G, H, N = sp.symbols('x A B F G H N', real=True)
Q = sp.symbols('Q', real=True, positive=True)


# Define symbolic perturbation characters and their corresponding sympy symbols
symnoise_char_list = ['A', 'B', 'F', 'G', 'H']
symnoise_sym_list = [A, B, F, G, H]
local_sym_dict = {'x': x, 'A': A, 'B': B, 'F': F, 'G': G, 'H': H, 'N': N, 'Q': Q}

# These sources contain explicit hypergeometric functions, which are marked by 'F' - so 'F' is not treated as a symbolic perturbation character. 
# The hg_ variables below represent this special treatment.
hypergeomatric_question_sources = ["ASyMOB\nHypergeometrics\nQ1", "ASyMOB\nHypergeometrics\nQ2", "ASyMOB\nHypergeometrics\nQ3", "ASyMOB\nHypergeometrics\nQ4"]
hg_symnoise_char_list = ['A', 'B', 'G', 'H']
hg_symnoise_sym_list = [A, B, G, H]
hg_local_sym_dict = {'x': x, 'A': A, 'B': B, 'G': G, 'H': H, 'N': N, 'Q': Q}


# List of easy equivalent forms (should simplify to 1)
equivalent_forms_easy = [
    sp.sin(-Q*x)**2 + sp.cos(Q*x)**2,
    -sp.sinh(Q*x)**2 + sp.cosh(Q*x)**2,
    (sp.log(x) * sp.log(Q,x))/sp.log(Q),
    Q * sp.Sum( x / (Q * 2**N) , (N, 1, sp.oo)) / x,
    (sp.exp(sp.I * Q * x) - sp.exp(-sp.I * Q * x)) / (2 * sp.I * sp.sin(Q * x))
]
# List of hard equivalent forms (should simplify to 1)
equivalent_forms_hard = [
    (sp.tan((Q-1)*x) + sp.tan(x)) / ((1 - sp.tan((Q-1)*x) * sp.tan(x)) * sp.tan(Q*x)),
    sp.sinh(sp.log(Q*x + sp.sqrt((Q*x)**2 + 1))) / (Q*x),
    (sp.log(x / sp.E, Q) + sp.log(sp.E, Q))/ sp.log(x, Q),
    Q * sp.Sum( (6 * x) / (Q * (N * sp.pi)**2) , (N, 1, sp.oo)) / x,
    -((1 + sp.exp(4 * sp.I * Q * x)) / (1 - sp.exp(4 * sp.I * Q * x))) * (2 * sp.tan(Q*x) / ((1 - sp.tan(Q*x)**2)) * sp.I)
]

# Test that all equivalent forms simplify to 1 and are numerically close to 1.
# Note that some expressions above do not simplify to 1 by sp.simplify - due to the CAS's limitations - but are evaluated correctly to 1 numerically.
# We still print the warning to raise user awareness.
equivalence_test_x = -2.5
equivalence_test_Q = 0.5
equivalence_test_margin = 1e-4
for form in (equivalent_forms_easy + equivalent_forms_hard):
    # Check if the form is equivalent to 1
    if sp.simplify(form) != 1 or (abs(form.subs(Q, equivalence_test_Q).subs(x, equivalence_test_x).evalf() - 1) > equivalence_test_margin):
        print(f"Form {form} is not equivalent to 1")
        print(f"{form} is simplified to {sp.simplify(form)}")
        print("Form is numerically evaluated to: ", form.subs(Q, equivalence_test_Q).subs(x, equivalence_test_x).evalf())
    
# LaTeX representations of the easy and hard equivalent forms
eq_forms_latex_easy = [
 r'\sin^{2}{\left(- Q x \right)} + \cos^{2}{\left(Q x \right)}',
 r'- \sinh^{2}{\left(Q x \right)} + \cosh^{2}{\left(Q x \right)}',
 r'\frac{\ln(x) \cdot \log_{x}(Q)}{\ln(Q)}',
 r'\frac{Q \sum_{N=1}^{\infty} \frac{2^{- N} x}{Q}}{x}',
 r'- \frac{i \left(e^{i Q x} - e^{- i Q x}\right)}{2 \sin{\left(Q x \right)}}']
eq_forms_latex_hard = [
 r'\frac{\tan{\left(x \right)} + \tan{\left(x \left(Q - 1\right) \right)}}{\left(- \tan{\left(x \right)} \tan{\left(x \left(Q - 1\right) \right)} + 1\right) \tan{\left(Q x \right)}}',
 r'\frac{\sinh{\left(\log{\left(Q x + \sqrt{Q^{2} x^{2} + 1} \right)} \right)}}{Q x}',
 r'\frac{\log_Q\left(\frac{x}{e}\right) + \log_Q(e)}{\log_Q(x)}',
 r'\frac{Q \sum_{N=1}^{\infty} \frac{6 x}{\pi^{2} N^{2} Q}}{x}',
 r'- \frac{2 i \left(e^{4 i Q x} + 1\right) \tan{\left(Q x \right)}}{\left(1 - e^{4 i Q x}\right) \left(1 - \tan^{2}{\left(Q x \right)}\right)}']

def replace_in_dollars(s, old, new):
    # Replace 'old' with 'new' inside all $...$ math substrings in s
    def repl(match):
        return match.group(0).replace(old, new)
    return re.sub(r'\$(.*?)\$', repl, s)

def char_in_dollars(s, char):
    """Return True if char appears inside any $...$ substring in s."""
    matches = re.findall(r'\$(.*?)\$', s)
    return any(char in match for match in matches)

def check_for_problematic_symbols(sp_ans):
    """Check for problematic symbols in sp_ans, but allow infinities if they are only used as summation or integration limits."""
    # Check for NaN or zoo anywhere
    if sp_ans.has(sp.nan) or sp_ans.has(sp.zoo):
        return True
    # Check for oo or -oo not as summation/integration limits
    def has_bad_infinity(expr):
        # If it's a Sum or Integral, skip limits
        if isinstance(expr, (sp.Sum, sp.Integral)):
            # expr.limits is a tuple of tuples: (symbol, lower, upper)
            # Only check the function part, not the limits
            return has_bad_infinity(expr.function)
        # If it's an infinity itself, it's problematic
        if expr == sp.oo or expr == -sp.oo:
            return True
        # Recursively check args
        return any(has_bad_infinity(arg) for arg in getattr(expr, 'args', []))
    return has_bad_infinity(sp_ans)

# Generate symbolic and numeric variants for each item in the dataset
def generate_variants(items, symnoise_chars, symnoise_syms, sym_dict, cur_ind):
    next_ind = cur_ind
    new_items = []
    for item in items:
        latex_chall = item.get("Challenge")
        sp_sym_ans = sp.sympify(item.get("Answer in Sympy"), locals = sym_dict)
        source = item.get("Source")
        chars_in_latex = []
        syms_in_latex = []
        # Find which symbolic perturbation characters are present in the LaTeX challenge
        for i in range(len(symnoise_chars)):
            if char_in_dollars(latex_chall, symnoise_chars[i]):
                chars_in_latex.append(symnoise_chars[i])
                syms_in_latex.append(symnoise_syms[i])
        if len(chars_in_latex) == 0:
            print("No symbolic parameters found inside math expressions in source: ",source)

        item['Variation'] = f"Symbolic-{len(chars_in_latex)}"
            
        # Replace all symbols in symnoise_sym_list by 1 for equivalence perturbation answers.
        sp_sym_ans_ones = sp_sym_ans.subs(dict(zip(symnoise_syms, [1]*len(symnoise_syms))))
        # Check for problematic symbols in sp_sym_ans_ones
        if check_for_problematic_symbols(sp_sym_ans_ones):
            print(f"Warning: sp_sym_ans_ones for {item} contains problematic symbol(s): {sp_sym_ans_ones}")

        # Generate all permutations of easy/hard equivalent forms for all symbolic chars
        ordered_sets = list(itertools.permutations(range(len(eq_forms_latex_easy)), len(chars_in_latex)))
        for order in ordered_sets:
            # Substitute easy forms
            latex_chall_copy = latex_chall
            for i in range(len(chars_in_latex)):
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], r' \left(' + eq_forms_latex_easy[order[i]].replace('Q', chars_in_latex[i]) + r'\right) ' )
            next_ind += 1
            new_items.append({
                    "Index": str(next_ind),
                    "Challenge": latex_chall_copy,
                    "Answer in Sympy": str(sp_sym_ans_ones),
                    "Answer in Latex": "",
                    "Variation": "Equivalence-All-Easy",
                    "Source": source
                })
            # Substitute hard forms
            latex_chall_copy = latex_chall
            for i in range(len(chars_in_latex)):
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], r' \left(' + eq_forms_latex_hard[order[i]].replace('Q', chars_in_latex[i]) + r'\right) ' )
            next_ind += 1
            new_items.append({
                    "Index": str(next_ind),
                    "Challenge": latex_chall_copy,
                    "Answer in Sympy": str(sp_sym_ans_ones),
                    "Answer in Latex": "",
                    "Variation": "Equivalence-All-Hard",
                    "Source": source
                })

        # Generate single-symbolic substitutions (easy/hard) and numeric perturbation variants
        for i in range(len(chars_in_latex)):
            chars_left_in_latex = chars_in_latex.copy()
            chars_left_in_latex.pop(i)
            for j in range(len(eq_forms_latex_easy)):
                # Substitute one easy form
                latex_chall_copy = latex_chall
                replace_form = r' \left(' + eq_forms_latex_easy[j].replace('Q', chars_in_latex[i]) + r'\right) '
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], replace_form )
                for ch in chars_left_in_latex:
                    latex_chall_copy = replace_in_dollars(latex_chall_copy, ch, '')
                next_ind += 1
                new_items.append({
                        "Index": str(next_ind),
                        "Challenge": latex_chall_copy,
                        "Answer in Sympy": str(sp_sym_ans_ones),
                        "Answer in Latex": "",
                        "Variation": "Equivalence-One-Easy",
                        "Source": source
                    })
                # Substitute one hard form
                latex_chall_copy = latex_chall
                replace_form = r' \left(' + eq_forms_latex_hard[j].replace('Q', chars_in_latex[i]) + r'\right) '
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], replace_form)
                for ch in chars_left_in_latex:
                    latex_chall_copy = replace_in_dollars(latex_chall_copy, ch, '')
                next_ind += 1
                new_items.append({
                        "Index": str(next_ind),
                        "Challenge": latex_chall_copy,
                        "Answer in Sympy": str(sp_sym_ans_ones),
                        "Answer in Latex": "",
                        "Variation": "Equivalence-One-Hard",
                        "Source": source
                    })
                
            # Numeric perturbation: replace one symbol with a random integer of increasing digit length
            for noise_digits in range(1, 11):
                latex_chall_copy = latex_chall
                sp_sym_ans_copy = sp_sym_ans
                nn1 = random.randint(10**(noise_digits-1), 10**noise_digits - 1)
                sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[i], sp.UnevaluatedExpr(nn1), evaluate=False)
                while check_for_problematic_symbols(sp_sym_ans_copy):
                    print(f"Warning: Numeric-One noise for {item} contains problems: {sp_sym_ans_copy}. Retrying")
                    nn1 = random.randint(10**(noise_digits-1), 10**noise_digits - 1)
                    sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[i], sp.UnevaluatedExpr(nn1), evaluate=False)
                replace_form = r' \left(' + str(nn1) + r'\right) '
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], replace_form)
                for ch in chars_left_in_latex:
                    latex_chall_copy = replace_in_dollars(latex_chall_copy, ch, '')
                for sym in syms_in_latex:
                    sp_sym_ans_copy = sp_sym_ans_copy.subs(sym, 1)
                latex_chall_copy = re.sub(r'Assume.*?\.', '', latex_chall_copy)
                next_ind += 1
                new_items.append({
                    "Index": str(next_ind),
                    "Challenge": latex_chall_copy,
                    "Answer in Sympy": str(sp_sym_ans_copy),
                    "Answer in Latex": "",
                    "Variation": f"Numeric-One-{noise_digits}",
                    "Source": source
                })

            
        # Numeric perturbation: replace all symbols with random integers of increasing digit length
        for noise_digits in range(1, 11):
            latex_chall_copy = latex_chall
            sp_sym_ans_copy = sp_sym_ans
            nn_lst = [random.randint(10**(noise_digits-1), 10**noise_digits - 1) for _ in range(len(chars_in_latex))]
            for i in range(len(chars_in_latex)):
                sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[i], sp.UnevaluatedExpr(nn_lst[i]), evaluate=False)
            while check_for_problematic_symbols(sp_sym_ans_copy):
                print(f"Warning: Numeric-All noise for {item} contains problems: {sp_sym_ans_copy}. Retrying")
                nn_lst = [random.randint(10**(noise_digits-1), 10**noise_digits - 1) for _ in range(len(chars_in_latex))]
                for i in range(len(chars_in_latex)):
                    sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[i], sp.UnevaluatedExpr(nn_lst[i]), evaluate=False)
            for i in range(len(chars_in_latex)):
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], r' \left(' + str(nn_lst[i]) + r'\right) ' )

            # Remove "Assume ... ." clause from latex_chall if it exists
            latex_chall_copy = re.sub(r'Assume.*?\.', '', latex_chall_copy)

            # Add new item with rolling index
            next_ind += 1
            new_items.append({
                "Index": str(next_ind),
                "Challenge": latex_chall_copy,
                "Answer in Sympy": str(sp_sym_ans_copy),
                "Answer in Latex": "",
                "Variation": f"Numeric-All-{noise_digits}",
                "Source": source
            })
            
        # Generate variants with some symbols replaced by 1 (partial symbolic)
        for i in range(1, len(chars_in_latex)):
            oned_indexes = list(itertools.combinations(range(len(chars_in_latex)), i))
            for oned_set in oned_indexes:
                latex_chall_copy = latex_chall
                sp_sym_ans_copy = sp_sym_ans
                for ind in oned_set:
                    latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[ind], '')
                    sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[ind], 1)
                next_ind += 1
                new_items.append({
                    "Index": str(next_ind),
                    "Challenge": latex_chall_copy,
                    "Answer in Sympy": str(sp_sym_ans_copy),
                    "Answer in Latex": "",
                    "Variation": f"Symbolic-{len(chars_in_latex) - i}",
                    "Source": source
                })
    # Add new_items to data before writing output
    data.extend(new_items)

# Generate 'Numeric-All-2-S' variants - the 'Variance' subset
def generate_NA2S(items, symnoise_chars, symnoise_syms, sym_dict, cur_ind, noise_digits, reps_num):
    next_ind = cur_ind
    new_items = []

    for item in items:
        latex_chall = item.get("Challenge")
        sp_sym_ans = sp.sympify(item.get("Answer in Sympy"), locals = sym_dict)
        source = item.get("Source")
        chars_in_latex = []
        syms_in_latex = []
        # Find which symbolic noise characters are present in the LaTeX challenge
        for i in range(len(symnoise_chars)):
            if char_in_dollars(latex_chall, symnoise_chars[i]):
                chars_in_latex.append(symnoise_chars[i])
                syms_in_latex.append(symnoise_syms[i])
        if len(chars_in_latex) == 0:
            print("No symbolic parameters found inside math expressions in source: ",source)
        
        for _ in range(reps_num):
            latex_chall_copy = latex_chall
            sp_sym_ans_copy = sp_sym_ans
            # Generate random integer values for all symbolic chars
            nn_lst = [random.randint(10**(noise_digits-1), 10**(noise_digits) - 1) for _ in range(len(chars_in_latex))]
            for i in range(len(chars_in_latex)):
                sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[i], sp.UnevaluatedExpr(nn_lst[i]), evaluate=False)

            while check_for_problematic_symbols(sp_sym_ans_copy):
                print(f"Warning: Numeric-All noise for {item} contains problems: {sp_sym_ans_copy}. Retrying")
                nn_lst = [random.randint(10**(noise_digits-1), 10**noise_digits - 1) for _ in range(len(chars_in_latex))]
                for i in range(len(chars_in_latex)):
                    sp_sym_ans_copy = sp_sym_ans_copy.subs(syms_in_latex[i], sp.UnevaluatedExpr(nn_lst[i]), evaluate=False)

            for i in range(len(chars_in_latex)):
                latex_chall_copy = replace_in_dollars(latex_chall_copy, chars_in_latex[i], r' \left(' + str(nn_lst[i]) + r'\right) ' )
                
            # Remove "Assume ... ." clause from latex_chall if it exists
            latex_chall_copy = re.sub(r'Assume.*?\.', '', latex_chall_copy)
            next_ind += 1
            new_items.append({
                "Index": str(next_ind),
                "Challenge": latex_chall_copy,
                "Answer in Sympy": str(sp_sym_ans_copy),
                "Answer in Latex": "",
                "Variation": f"Numeric-All-{noise_digits}-S",
                "Source": source
            })
    # Add new_items to data before writing output
    data.extend(new_items)


# Split items into regular and hypergeometric symbolic questions
sym_var_items = [item for item in data if (item.get('Variation', '').strip() == 'Symbolic' and item.get('Source') not in hypergeomatric_question_sources)]
hypergeometric_sym_var_items = [item for item in data if (item.get('Variation', '').strip() == 'Symbolic' and item.get('Source') in hypergeomatric_question_sources)]

# Generate all variants for regular and hypergeometric items
generate_variants(sym_var_items, symnoise_char_list, symnoise_sym_list, local_sym_dict, cur_data_len)
cur_data_len = len(data)
generate_variants(hypergeometric_sym_var_items, hg_symnoise_char_list, hg_symnoise_sym_list, hg_local_sym_dict, cur_data_len)
cur_data_len = len(data)
generate_NA2S(sym_var_items, symnoise_char_list, symnoise_sym_list, local_sym_dict, cur_data_len, 2, 50)
cur_data_len = len(data)
generate_NA2S(hypergeometric_sym_var_items, hg_symnoise_char_list, hg_symnoise_sym_list, hg_local_sym_dict, cur_data_len, 2, 50)
cur_data_len = len(data)
print("Final size of the ASyMOB dataset is: " ,cur_data_len) 

# Write the full dataset to a JSON file
output_json_path = 'Full_ASyMOB_Dataset.json'
with open(output_json_path, 'w', encoding='utf-8') as jf:
    json.dump(data, jf, ensure_ascii=False, indent=2)

