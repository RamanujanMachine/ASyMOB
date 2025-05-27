create table challenges
(
    challenge_id integer not null
        primary key,
    challenge    text,
    answer_sympy text,
    answer_latex text,
    variation    text,
    source       text
);

create table model_responses
(
    response_id        uuid default gen_random_uuid() not null
        primary key,
    challenge_id       integer,
    model              text,
    code_execution     boolean,
    prompt             text,
    full_answer        text,
    tokens_used        integer,
    final_answer_latex text,
    error              text,
    acquisition_method text,
    acquisition_time   timestamp
);

create index model_responses_challenge_id_index
    on model_responses (challenge_id);

create index model_responses_model_index
    on model_responses (model);

create index model_responses_code_execution_index
    on model_responses (code_execution);

create table numeric_verification
(
    response_id              uuid not null
        constraint numeric_varification_pkey
            primary key,
    numeric_correct          boolean,
    strict_mode              boolean,
    numeric_comparison_error text,
    check_time               timestamp,
    latex_parsing_method     text,
    model_answer_sympy       text
);

create table numerical_substitutions
(
    challenge_id    integer,
    subs_json       text,
    numerical_value text
);

create table symbolic_verification
(
    response_id               uuid not null
        constraint symbolic_varification_pkey
            primary key,
    symbolic_correct          boolean,
    symbolic_comparison_error text,
    check_time                timestamp,
    latex_parsing_method      text,
    model_answer_sympy        text
);

create view pipeline_results_extra
            (challenge_id, variation, source, true_answer_sympy, response_id, model, code_execution, full_answer,
             final_answer_latex, acquisition_time, error, model_answer, symbolic_correct, numeric_correct,
             symbolic_comparison_error, numeric_comparison_error, response_acquisition_time, symbolic_check_time,
             numeric_check_time, nv_model_answer_used, sv_model_answer_used, nv_latex_parsing_method,
             sv_latex_parsing_method)
as
SELECT qs.challenge_id,
       qs.variation,
       qs.source,
       qs.answer_sympy         AS true_answer_sympy,
       mr.response_id,
       mr.model,
       mr.code_execution,
       mr.full_answer,
       mr.final_answer_latex,
       mr.acquisition_time,
       mr.error,
       sv.model_answer_sympy   AS model_answer,
       sv.symbolic_correct,
       nv.numeric_correct,
       sv.symbolic_comparison_error,
       nv.numeric_comparison_error,
       mr.acquisition_time     AS response_acquisition_time,
       sv.check_time           AS symbolic_check_time,
       nv.check_time           AS numeric_check_time,
       nv.model_answer_sympy   AS nv_model_answer_used,
       sv.model_answer_sympy   AS sv_model_answer_used,
       nv.latex_parsing_method AS nv_latex_parsing_method,
       sv.latex_parsing_method AS sv_latex_parsing_method
FROM asymob.challenges qs
         LEFT JOIN asymob.model_responses mr USING (challenge_id)
         LEFT JOIN asymob.numeric_verification nv USING (response_id)
         LEFT JOIN asymob.symbolic_verification sv USING (response_id)
WHERE qs.challenge_id < 17092;


create view pipeline_results_classified
            (challenge_id, variation, source, true_answer_sympy, response_id, model, code_execution, full_answer,
             final_answer_latex, acquisition_time, error, tokens_used, model_answer, symbolic_correct, numeric_correct,
             symbolic_comparison_error, numeric_comparison_error, response_acquisition_time, symbolic_check_time,
             numeric_check_time, nv_model_answer_used, sv_model_answer_used, nv_latex_parsing_method,
             sv_latex_parsing_method, state)
as
WITH pipeline AS (SELECT qs.challenge_id,
                         qs.variation,
                         qs.source,
                         qs.answer_sympy         AS true_answer_sympy,
                         mr.response_id,
                         mr.model,
                         mr.code_execution,
                         mr.full_answer,
                         mr.final_answer_latex,
                         mr.acquisition_time,
                         mr.error,
                         mr.tokens_used,
                         sv.model_answer_sympy   AS model_answer,
                         sv.symbolic_correct,
                         nv.numeric_correct,
                         sv.symbolic_comparison_error,
                         nv.numeric_comparison_error,
                         mr.acquisition_time     AS response_acquisition_time,
                         sv.check_time           AS symbolic_check_time,
                         nv.check_time           AS numeric_check_time,
                         nv.model_answer_sympy   AS nv_model_answer_used,
                         sv.model_answer_sympy   AS sv_model_answer_used,
                         nv.latex_parsing_method AS nv_latex_parsing_method,
                         sv.latex_parsing_method AS sv_latex_parsing_method
                  FROM asymob.challenges qs
                           LEFT JOIN asymob.model_responses mr USING (challenge_id)
                           LEFT JOIN asymob.numeric_verification nv USING (response_id)
                           LEFT JOIN asymob.symbolic_verification sv USING (response_id)
                  WHERE qs.challenge_id < 17092),
     tab AS (SELECT pipeline.challenge_id,
                    pipeline.variation,
                    pipeline.source,
                    pipeline.true_answer_sympy,
                    pipeline.response_id,
                    pipeline.model,
                    pipeline.code_execution,
                    pipeline.full_answer,
                    pipeline.final_answer_latex,
                    pipeline.acquisition_time,
                    pipeline.error,
                    pipeline.tokens_used,
                    pipeline.model_answer,
                    pipeline.symbolic_correct,
                    pipeline.numeric_correct,
                    pipeline.symbolic_comparison_error,
                    pipeline.numeric_comparison_error,
                    pipeline.response_acquisition_time,
                    pipeline.symbolic_check_time,
                    pipeline.numeric_check_time,
                    pipeline.nv_model_answer_used,
                    pipeline.sv_model_answer_used,
                    pipeline.nv_latex_parsing_method,
                    pipeline.sv_latex_parsing_method,
                    CASE
                        WHEN pipeline.numeric_correct IS NOT NULL OR pipeline.symbolic_correct IS NOT NULL THEN 'passed'::text
                        WHEN pipeline.full_answer IS NULL AND pipeline.error IS NOT NULL AND
                             (pipeline.error ~~ '%timeout%'::text OR pipeline.error ~~ '%Time%'::text)
                            THEN 'llm timeout'::text
                        WHEN pipeline.full_answer IS NULL THEN 'failed to get answer'::text
                        WHEN pipeline.error ~~ '%No latex answer found in the textual answer.%'::text
                            THEN 'latex not found'::text
                        WHEN pipeline.final_answer_latex IS NULL THEN 'latex extract failed'::text
                        WHEN pipeline.symbolic_comparison_error = 'timeout'::text AND
                             pipeline.numeric_comparison_error = 'timeout'::text THEN 'double comparison timeout'::text
                        WHEN (pipeline.nv_model_answer_used IS NULL OR pipeline.nv_model_answer_used = 'None'::text) AND
                             (pipeline.sv_model_answer_used IS NULL OR pipeline.sv_model_answer_used = 'None'::text) AND
                             pipeline.error IS NULL AND pipeline.symbolic_comparison_error IS NULL AND
                             pipeline.numeric_comparison_error IS NULL
                            THEN 'no sympy data; no exception (unknown state)'::text
                        WHEN (pipeline.nv_model_answer_used IS NULL OR pipeline.nv_model_answer_used = 'None'::text) AND
                             (pipeline.sv_model_answer_used IS NULL OR pipeline.sv_model_answer_used = 'None'::text) AND
                             NOT (pipeline.error IS NULL AND pipeline.symbolic_comparison_error IS NULL AND
                                  pipeline.numeric_comparison_error IS NULL) THEN 'no sympy data; with exception'::text
                        ELSE NULL::text
                        END AS state
             FROM pipeline)
SELECT challenge_id,
       variation,
       source,
       true_answer_sympy,
       response_id,
       model,
       code_execution,
       full_answer,
       final_answer_latex,
       acquisition_time,
       error,
       tokens_used,
       model_answer,
       symbolic_correct,
       numeric_correct,
       symbolic_comparison_error,
       numeric_comparison_error,
       response_acquisition_time,
       symbolic_check_time,
       numeric_check_time,
       nv_model_answer_used,
       sv_model_answer_used,
       nv_latex_parsing_method,
       sv_latex_parsing_method,
       state
FROM tab;


create view asymob.pipeline_results as
select
    qs.challenge_id,
    qs.variation,
    qs.source,
    qs.answer_sympy as true_answer_sympy,
    mr.model,
    mr.code_execution,
    mr.full_answer,
    mr.final_answer_latex,
    mr.tokens_used,
    sv.symbolic_correct,
    nv.numeric_correct

from asymob.challenges qs
    left join asymob.model_responses mr
        using (challenge_id)
    left join asymob.numeric_verification nv
        using (response_id)
    left join asymob.symbolic_verification sv
        using (response_id)
where challenge_id < 17092
