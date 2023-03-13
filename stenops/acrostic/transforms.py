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


def _split_numbered_list(inputs):
    text = inputs["generated_sentence_options"]
    sentences = re.split(r"(?<=[.?!])\s+(?=[0-9])", text)
    sentences = [s.strip() for s in sentences if len(s) > 0]
    sentences = [re.sub(r"^[0-9]+\.\s*", "", s) for s in sentences]

    return {"generated_sentence_options_list": sentences}


def _split_bulleted_list(inputs):
    text = inputs["original_text_summary"]
    # split on newlines
    sentences = [s for s in text.split("\n") if len(s) > 0]

    return {"original_text_summary_list": sentences}


def _get_current_idea(inputs):
    acrostic_letter_index = inputs["acrostic_letter_index"]
    idea = inputs["original_text_summary_list"]
    return {"current_idea": idea[acrostic_letter_index]}


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = (
    "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
)
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"


def _split_into_sentences(inputs):
    text = inputs["generated_sentence_options"]
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
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

get_current_idea = TransformChain(
    input_variables=["original_text_summary_list", "acrostic_letter_index"],
    transform=_get_current_idea,
    output_variables=["current_idea"],
)

split_numbered_list = TransformChain(
    input_variables=["generated_sentence_options"],
    transform=_split_numbered_list,
    output_variables=["generated_sentence_options_list"],
)

split_bulleted_list = TransformChain(
    input_variables=["original_text_summary"],
    transform=_split_bulleted_list,
    output_variables=["original_text_summary_list"],
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
