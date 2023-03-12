from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI

from stegosaurus.acrostic import transforms as t
from stegosaurus.acrostic.prompts import (
    acrostic_evaluator_prompt,
    multi_acrostic_generator_prompt,
)

load_dotenv()

acrostic_generator_chain = LLMChain(
    llm=OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=1,
    ),
    prompt=multi_acrostic_generator_prompt,
    output_key="generated_sentence_options",
)


acrostic_evaluator_chain = LLMChain(
    llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
    prompt=acrostic_evaluator_prompt,
    output_key="evaluator_output",
)


full_chain = SequentialChain(
    chains=[
        t.get_current_acrostic_letter,
        t.get_current_original_sentence,
        acrostic_generator_chain,
        t.split_into_sentences,
        t.generate_clean_options,
        acrostic_evaluator_chain,
        t.extract_selection,
        t.resolve_next_action,
    ],
    input_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "acrostic_letter_index",
        "n_attempts",
    ],
    output_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "acrostic_letter_index",
        "n_attempts",
        "current_starting_letter",
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
