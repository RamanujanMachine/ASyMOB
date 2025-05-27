# ASyMOB Dataset Generation

ASyMOB_Generation.py generates a diverse set of mathematical question variants from a seed CSV file. It leverages the `sympy` library for symbolic mathematics to create various perturbations of original questions, including symbolic, numeric, and equivalence-based transformations. The generated questions are then saved to a JSON file.

## Prerequisites

- Python 3.7+
- `sympy` library (`pip install sympy`)
- csv, json, re, random, itertools

## Usage

1.  **Prepare your seed data:** Ensure you have a CSV file named `Seed_and_Max_Symbolic_Perturbations.csv` in the same directory as the script. This CSV should contain the seed mathematical questions, their maximal symbolic perturbations, and answers as SymPy expressions.

    The expected fields in `Seed_and_Max_Symbolic_Perturbations.csv` are:
    * **`Challenge`**: The mathematical question in LaTeX format, including assumptions regarding variables or other mathematical details.
    * **`Answer in LaTeX`** (optional): The answer to the question, represented as a LaTeX string.
    * **`Answer in Sympy`**: The answer to the question, represented as a SymPy expression string.
    * **`Variation`**: "Original" or "Symbolic".
    * **`Source`**: Identifies the origin of the question.

2.  **Run the script:**
    ```
    python ASyMOB_Generation.py
    ```

3.  **Output:** The script will generate a JSON file named `Full_ASyMOB_Dataset.json` in the same directory. This file will contain all the original seed and symbolic questions along with their newly generated symbolic, numeric, and equivalence-based variants.

    The resulting fields in `Full_ASyMOB_Dataset.json` are:
    * **`Index`**: Sequential ID.
    * **`Challenge`**: The mathematical question in LaTeX format, including assumptions regarding variables or other mathematical details.
    * **`Answer in Sympy`**: The answer to the question, represented as a SymPy expression string.
    * **`Variation`**: e.g., Equivalence-All-Hard, Numeric-One-3, etc.
    * **`Source`**: Same as the seed question from which this variation originated.

## Customization

-   **`Seed_and_Max_Symbolic_Perturbations.csv`:** Modify this CSV to add new seed questions or adjust existing ones.
-   **`symnoise_char_list`, `symnoise_sym_list`:** Adjust the lists of symbolic characters and their SymPy representations if your questions use different symbols for perturbation (ASyMOB uses 'A', 'B', 'F', 'G', 'H' by default).
-   **`equivalent_forms_easy`, `equivalent_forms_hard`:** Add or modify the equivalent forms to introduce different types of mathematical equivalences.
-   **`noise_digits` and `reps_num`:** In `generate_NA2S`, you can change `noise_digits` to control the range of random numbers used for numeric perturbations and `reps_num` to control the number of repetitions for each item.

# ASyMOB Evaluation Pipeline

The ASyMOB evaluation pipeline is a set of Python scripts that query different LLMs to solve the generated questions, then validate the answers both symbolically and numerically. The pipeline consists of the following scripts:
- **Answer collection scripts**: These scripts collect answers from the LLMs for the questions generated in the previous step, using different APIs.
    - `collect_llm_answers.py`: Uses the Responses or Completions API to query LLMs one question at a time. Implementations for different LLMs are provided in the `interface` files.
    - `openai_send_batch_query.py`: Uses the BATCH API to query OpenAI's models in bulk. This implementation reduces execution time and costs.
    - `openai_assistants.py`: Uses the OpenAI Assistant API for querying. This allows forcing the LLM to spawn a Python interpreter (server side) and use it for question solving.
- **Answer validation script**: `validate_answers_rowwise.py` validates the collected answers using symbolic and numerical validation. Numeric validation relies on numerical substitution of the variables in the answers. A set of valid substitutions is generated using the `create_subs.py` script. Symbolic validation relies on the SymPy library to validate the answers symbolically.

## Usage
The evaluation pipeline maintains a database of the answers collected and their validations. The database's SQL definition is provided in the `db_schema.sql` file.

Sample tables are provided in the `sample_data` folder. You can use these samples by uncommenting the relevant lines in the respective files.

To collect the answers, run:
```
python collect_llm_answers.py
```

To validate the answers, run:
```
python validate_answers_rowwise.py
```

## OUTPUT
The output of the validation pipeline is stored in a database. The pipeline's 
results are stored in the view `pipeline_results`. The columns of the view are:
- *challenge_id* - The ID of the challenge in the benchmark dataset.
- *variation* - The variation type of the challenge, e.g., "Numeric-One-3", "Symbolic-All".
- *source* - The source of the challenge.
- *true_answer_sympy* - The correct answer to the challenge in SymPy format.
- *model* - The model used to answer the challenge, e.g., "o4-mini", "gpt-4o".
- *code_execution* - Indicates whether the prompt used incentivized the model to execute code or disallowed it. `None` indicates neither.
- *full_answer* - The full answer provided by the model, including any code execution output.
- *final_answer_latex* - The final answer provided by the model in LaTeX format, extracted from the full answer.
- *tokens_used* - The number of tokens used by the model to answer the query.
- *symbolic_correct* - Whether the answer was successfully validated symbolically.
- *numeric_correc* - Whether the answer was successfully validated numerically.
