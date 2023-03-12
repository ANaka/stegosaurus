import json
import re

from langchain.chains import TransformChain


def _get_current_acrostic_letter(inputs):
    acrostic_phrase = inputs["acrostic_phrase"]
    acrostic_letter_index = inputs["acrostic_letter_index"]
    acrostic_letters = [letter for letter in acrostic_phrase]
    current_starting_letter = acrostic_letters[acrostic_letter_index]
    return {"current_starting_letter": current_starting_letter}


def _get_current_original_sentence(inputs):
    acrostic_letter_index = inputs["acrostic_letter_index"]
    sentences = [s for s in inputs["original_text"].split(".") if len(s) > 0]
    return {"current_original_sentence": sentences[acrostic_letter_index]}


def _split_into_sentences(inputs):
    text = inputs["generated_sentence_options"]
    sentences = re.split(r"(?<=[.?!])\s+(?=[0-9])", text)
    sentences = [s.strip() for s in sentences]
    sentences = [re.sub(r"^[0-9]+\.\s*", "", s) for s in sentences]

    return {"generated_sentence_options_list": sentences}


def _generate_clean_options(inputs):
    generated_sentence_options_list = inputs["generated_sentence_options_list"]
    output_string = ""
    for i, sentence in enumerate(generated_sentence_options_list):
        output_string += f"{i+1}. {sentence}\n"
    output_string += f"{len(generated_sentence_options_list)+1}. None of these look good. AGA, try again."
    return {"clean_options": output_string}


def _extract_selection(inputs):
    selection = int(json.loads(inputs["evaluator_output"])["selected_option"])
    return {"evaluator_selection": selection}


def _resolve_next_action(inputs):

    try:
        assert isinstance(inputs["evaluator_selection"], int)
    except AssertionError:
        action = "regenerate (error)"
    if inputs["evaluator_selection"] == inputs["n_attempts"] + 1:
        action = "regenerate (requested)"
    if inputs["evaluator_selection"] <= inputs["n_attempts"]:
        action = "accept sentence"
    return {"next_action": action}


get_current_acrostic_letter = TransformChain(
    input_variables=["acrostic_phrase", "acrostic_letter_index"],
    transform=_get_current_acrostic_letter,
    output_variables=["current_starting_letter"],
)

get_current_original_sentence = TransformChain(
    input_variables=["original_text", "acrostic_letter_index"],
    transform=_get_current_original_sentence,
    output_variables=["current_original_sentence"],
)


split_into_sentences = TransformChain(
    input_variables=["generated_sentence_options"],
    transform=_split_into_sentences,
    output_variables=["generated_sentence_options_list"],
)

generate_clean_options = TransformChain(
    input_variables=["generated_sentence_options_list"],
    transform=_generate_clean_options,
    output_variables=["clean_options"],
)


extract_selection = TransformChain(
    input_variables=["evaluator_output"],
    transform=_extract_selection,
    output_variables=["evaluator_selection"],
)

resolve_next_action = TransformChain(
    input_variables=["evaluator_selection", "n_attempts"],
    transform=_resolve_next_action,
    output_variables=["next_action"],
)
