# import os
# from typing import Optional, Tuple

# import langchain
# from langchain.cache import InMemoryCache

# from stenops.acrostic.chains import get_full_chain

# langchain.llm_cache = InMemoryCache()
# from threading import Lock
# from pathlib import Path

import langchain

# from fastapi import FastAPI
import modal
from langchain.cache import InMemoryCache

from stenops.modal_image import image

langchain.llm_cache = InMemoryCache()

stub = modal.Stub(
    name="stenops",
    image=image,
    secrets=[
        modal.Secret.from_name("OPENAI_API_KEY"),
    ],
)


@stub.function
def parse_inputs(
    original_text: str,
    acrostic_phrase: str,
    rewritten_text: str = None,
    acrostic_letter_index: int = None,
    n_attempts: int = None,
):
    from stenops.acrostic.params import AcrosticGeneratorInputs

    inputs = AcrosticGeneratorInputs(
        original_text=original_text,
        acrostic_phrase=acrostic_phrase,
        rewritten_text=rewritten_text,
        acrostic_letter_index=acrostic_letter_index,
        n_attempts=n_attempts,
    )
    return inputs


@stub.function
def get_full_chain():
    from stenops.acrostic.chains import FullChain

    return FullChain()


@stub.function
def get_original_text_summarizer():
    from stenops.acrostic.chains import OriginalTextSummarizerChain

    return OriginalTextSummarizerChain()


@stub.function
def summarize_original_text(inputs):
    chain = get_original_text_summarizer()
    return chain(inputs=inputs.dict())


@stub.function
def validate_inputs(
    original_text: str,
    acrostic_phrase: str,
    rewritten_text: str = None,
    acrostic_letter_index: int = None,
    n_attempts: int = None,
):
    from stenops.acrostic.params import AcrosticGeneratorInputs

    return AcrosticGeneratorInputs(
        original_text=original_text,
        acrostic_phrase=acrostic_phrase,
        rewritten_text=rewritten_text,
        acrostic_letter_index=acrostic_letter_index,
        n_attempts=n_attempts,
    )


@stub.function
def do_one_round(
    original_text: str,
    acrostic_phrase: str,
    rewritten_text: str = None,
    acrostic_letter_index: int = None,
    n_attempts: int = None,
):
    inputs = parse_inputs(
        original_text=original_text,
        acrostic_phrase=acrostic_phrase,
        rewritten_text=rewritten_text,
        acrostic_letter_index=acrostic_letter_index,
        n_attempts=n_attempts,
    )

    output = summarize_original_text(inputs.dict())

    return output
    # chain = get_full_chain()
    # return chain(inputs=inputs.dict())


@stub.local_entrypoint
def main(
    original_text: str,
    acrostic_phrase: str,
):
    with stub.run():
        print(do_one_round.call(original_text=original_text, acrostic_phrase=acrostic_phrase))
