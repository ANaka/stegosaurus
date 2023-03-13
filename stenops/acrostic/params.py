from pydantic import BaseModel


class AcrosticGeneratorInputs(BaseModel):
    original_text: str
    acrostic_phrase: str
    rewritten_text: str = ""
    acrostic_letter_index: int = 0
    n_attempts: int = 5

    def __repr__(self) -> str:
        original_sentences = self.original_text.strip().split(".")
        original_sentences = [f"{s.strip()}." for s in original_sentences if s != ""]
        original_text_str = "\n".join(original_sentences)

        rewritten_sentences = self.rewritten_text.strip().split(".")
        rewritten_sentences = [f"{s.strip()}." for s in rewritten_sentences if s != ""]
        rewritten_text_str = "\n".join(rewritten_sentences)
        s = f"""
AcrosticGeneratorInputs

acrostic_phrase: {self.acrostic_phrase}

original_text:\n{original_text_str}

acrostic_letter_index: {self.acrostic_letter_index}
rewritten_text:\n{rewritten_text_str}

        """
        return s


class AcrosticGenerationRoundResult(BaseModel):
    round_number: int
    inputs: AcrosticGeneratorInputs
    output: dict
