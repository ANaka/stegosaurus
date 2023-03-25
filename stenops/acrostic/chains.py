from functools import partial

from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI

from stenops.acrostic import transforms as t
from stenops.acrostic.prompts import (
    acrostic_evaluator_prompt,
    multi_acrostic_generator_prompt,
    original_text_contextualizer_prompt,
    original_text_summarizer_prompt,
)

OriginalTextSummarizerChain = partial(
    LLMChain,
    llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
    prompt=original_text_summarizer_prompt,
    output_key="original_text_summary",
)

OriginalTextContextualizerChain = partial(
    LLMChain,
    llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
    prompt=original_text_contextualizer_prompt,
    output_key="original_text_context",
)

AcrosticGeneratorChain = partial(
    LLMChain,
    llm=OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
    ),
    prompt=multi_acrostic_generator_prompt,
    output_key="generated_sentence_options",
)

AcrosticEvaluatorChain = partial(
    LLMChain,
    llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
    prompt=acrostic_evaluator_prompt,
    output_key="evaluator_output",
)

GeneratorChain = partial(
    SequentialChain,
    chains=[
        OriginalTextSummarizerChain(),
        OriginalTextContextualizerChain(),
        t.split_bulleted_list,
        t.get_current_acrostic_letter,
        # t.get_current_idea,
        t.get_current_original_sentence,
        AcrosticGeneratorChain(),
        t.split_numbered_list,
        t.generate_clean_options,
    ],
    input_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "acrostic_letter_index",
        "n_attempts",
    ],
    output_variables=[
        "original_text_summary",
        "original_text_context",
        "current_starting_letter",
        # "current_idea",
        "current_original_sentence",
        "generated_sentence_options",
        "generated_sentence_options_list",
        "clean_options",
    ],
    verbose=False,
)

EvaluatorChain = partial(
    SequentialChain,
    chains=[
        AcrosticEvaluatorChain(),
        t.extract_selection,
        t.resolve_next_action,
    ],
    input_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "acrostic_letter_index",
        "n_attempts",
        "original_text_summary",
        "original_text_context",
        "current_starting_letter",
        "current_original_sentence",
        # "current_idea",
        "generated_sentence_options",
        "generated_sentence_options_list",
        "clean_options",
    ],
    output_variables=[
        "evaluator_output",
        "evaluator_selection",
        "next_action",
    ],
    verbose=False,
)


FullChain = partial(
    SequentialChain,
    chains=[
        GeneratorChain(),
        EvaluatorChain(),
    ],
    input_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "acrostic_letter_index",
        "n_attempts",
    ],
    output_variables=[
        "original_text_summary",
        "original_text_context",
        "current_starting_letter",
        # "current_idea",
        "current_original_sentence",
        "generated_sentence_options",
        "generated_sentence_options_list",
        "clean_options",
        "evaluator_output",
        "evaluator_selection",
        "next_action",
    ],
    verbose=False,
)
