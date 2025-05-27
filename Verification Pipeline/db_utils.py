from models import MODELS
from db_conn_parameters import CONN_PARAMETERS
import psycopg2
import pandas as pd
import sympy as sp
from functools import cache
from sp_vars import var_mapping

MODELS_LIST = MODELS.keys()


def get_connection():
    return psycopg2.connect(**CONN_PARAMETERS)


def _expand_models_list(models):
    """
    Expands a list of models to include the code on/ off versions of each model.
    """
    models_with_code = []
    for model in models:
        if isinstance(model, tuple):
            models_with_code.append(model)
            continue
        if not isinstance(model, str):
            raise ValueError(f"Model {model} is not a string or tuple")
        
        if MODELS[model].support_code():
            models_with_code.append((model, True))
            models_with_code.append((model, False))
        else:
            models_with_code.append((model, None))
    return models_with_code


def _get_model_filter(model, code):
    if code is None:
        return f"model = '{model}' and code_execution is null"
    else:
        return f"model = '{model}' and code_execution = {code}"


def load_tasks(models=MODELS_LIST, retry_errors=True, sql_filter=None,
                   parse_sympy=False):
    queries = []
    for model, code in _expand_models_list(models):
        queries.append((f"""
        with desired_model_responses as (
            select * from asymob.model_responses
            where {_get_model_filter(model, code)}
            {'and error is null' if not retry_errors else ''}
        )
        select challenge_id, challenge, answer_sympy, variation, source, error
        from asymob.challenges 
            left join desired_model_responses using (challenge_id)
        where full_answer is null 
        {'and ' + sql_filter if sql_filter else ''}
        """, model, code))
    
    query_columns = [
        'challenge_id', 'challenge', 'answer_sympy', 'variation', 'source', 
        'error'
    ]
    df = pd.DataFrame(columns=query_columns + ['model', 'code_execution'])
    with psycopg2.connect(**CONN_PARAMETERS) as conn:
        with conn.cursor() as cursor:
            for query, model, code in queries:
                cursor.execute(query)
                model_tasks = cursor.fetchall()
                model_tasks = pd.DataFrame(model_tasks, columns=query_columns)
                model_tasks['model'] = model
                model_tasks['code_execution'] = code
                df = pd.concat([df, model_tasks], ignore_index=True)
    df = df.reset_index(drop=True)

    if parse_sympy:
        @cache
        def _cached_parsing(sp_str):
            return sp.parse_expr(sp_str, evaluate=False, local_dict=var_mapping)
        df['answer_sympy'] = df['answer_sympy'].apply(_cached_parsing)
    
    return df

def insert_row(conn, table_name, data_dict):
    """
    Insert a single row into a PostgreSQL table.
    
    Args:
        conn: psycopg2 connection object (should not be shared across processes)
        table_name: str, name of the table
        data_dict: dict, column-value pairs to insert
    """
    columns = data_dict.keys()
    values = [data_dict[col] for col in columns]
    
    query = f"""
    INSERT INTO asymob.{table_name} ({', '.join(columns)})
    VALUES ({', '.join(['%s'] * len(values))})
    """
    
    with conn.cursor() as cur:
        cur.execute(query, values)
    conn.commit()
