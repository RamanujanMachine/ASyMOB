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

    Resulting fields in `Full_ASyMOB_Dataset.json` are:
    * **`Index`**: Sequential ID.
    * **`Challenge`** The mathematical question in LaTeX format, including assumptions regarding variables or other mathematical details.
    * **`Answer in Sympy`**: The answer to the question, represented as a SymPy expression string.
    * **`Variation`**: e.g., Equivalence-All-Hard, Numeric-One-3, etc.
    * **`Source`**: Same as the seed question from which this variation originated.

## Customization

-   **`Seed_and_Max_Symbolic_Perturbations.csv`:** Modify this CSV to add new seed questions or adjust existing ones.
-   **`symnoise_char_list`, `symnoise_sym_list`:** Adjust the lists of symbolic characters and their SymPy representations if your questions use different symbols for perturbation (ASyMOB uses 'A', 'B', 'F', 'G', 'H' by default).
-   **`equivalent_forms_easy`, `equivalent_forms_hard`:** Add or modify the equivalent forms to introduce different types of mathematical equivalences.
-   **`noise_digits` and `reps_num`:** In `generate_NA2S`, you can change `noise_digits` to control the range of random numbers used for numeric perturbations and `reps_num` to control the number of repetitions for each item.

# ASyMOB evaluation pipeline

ASyMOB evaluation pipeline are a set of Python script that asks different 
LLMs for the questions generated, and validates the answers symbolically and 
numerically.
The pipeline is composed of the following scripts:
- *Answer collection scripts:* These scripts collect the answers from the LLMs for the questions generated in the previous step, using different APIs.
    - `collect_llm_answers.py`: Using the Responses or Completions API to ask
    the LLMs the answers one-by-one. The implementation for different LLMs is
    provided in the `interface` files.
    - `openai_send_batch_query.py`: Using the BATHCH API to ask OpenAI's models 
    the answers in batch. This implementation reduces execution time and costs.
    - `openai_assistants.py`: Using the OpenAI's Assistant API to ask the LLMs the answers. Enables us to force the LLM to spawn a python interpreter and
    evaluate the answers using code.
- *Answer validation script:* `validate_answers_rowwise.py` - validates the
    answers collected from the LLMs using symbolic and numerical validation.
    The numeric validation relies on numerical substitution of the variables in
    the answers. A set of valid substitutions is provided is generated using 
    the `create_subs.py` script. The symbolic validation relies on the
    `sympy` library to validate the answers symbolically.

## Usage
The evaluation pipeline maintains a database of the answers collected and their
validations. The database's SQL definition is provided in the `db_schema.sql` file.

Samples for the tables are provided in sample_data folder. 
One can use those samples by uncommenting the lines for it in the respected files.

For collection the answers, run:
```
python collect_llm_answers.py
```

For validation, run:
```
python validate_answers_rowwise.py
```