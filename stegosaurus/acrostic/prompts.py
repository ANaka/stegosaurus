from langchain.prompts import PromptTemplate

acrostic_generator_prompt = PromptTemplate(
    input_variables=["original_text", "acrostic_phrase", "rewritten_text", "current_starting_letter"],
    template="""
You are an acrostic generating AI. I will give you some text and an acrostic phrase. Your job is to rewrite the text so that the first letter in each sentence spells out the acrostic phrase, while still preserving the meaning of the original text.

We'll do one sentence at a time. I'll give you the first letter of the new sentence, and your job will be to write a new sentence that starts with that letter. Then I'll check your work. If it looks good, I'll add it to the rewritten text and we'll move on to the next sentence. If not, I'll ask you to try again.

Original text: "{original_text}"
Acrostic phrase: "{acrostic_phrase}"
Rewritten text so far: "{rewritten_text}"
Starting letter for the next sentence: {current_starting_letter}

Now provide your answer. Remember that the first word must begin with the letter {current_starting_letter}.
""",
)

acrostic_evaluator_prompt = PromptTemplate(
    input_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "current_starting_letter",
        "clean_options",
    ],
    template="""
You are a helper to an acrostic generating AI named AGA.

I gave AGA instructions to rewrite some text so that the first letter in each sentence spells out an acrostic phrase, while still preserving the meaning of the original text.
AGA writes one sentence at a time and makes several guesses. Your job is to look at AGA's guesses and pick which one is the best.
A good sentence should start with the letter {current_starting_letter} and should also put us on track to end up with a rewritten text that captures the meaning of the original text while also sounding natural.

These are the inputs I gave AGA:
Original text: "{original_text}"
Acrostic phrase: "{acrostic_phrase}"
Rewritten text so far: "{rewritten_text}"
Starting letter for the next sentence: {current_starting_letter}

Now I'll give you AGA's attempts in multiple choice format. Your job is to pick the best one. If none of them look good, you can also indicate that and AGA will try again.

Provide your answer in the following format:
{{"selected_option": 1}}

Here are your options:
{clean_options}
""",
)

multi_acrostic_generator_prompt = PromptTemplate(
    input_variables=[
        "original_text",
        "acrostic_phrase",
        "rewritten_text",
        "current_starting_letter",
        "n_attempts",
        "current_original_sentence",
    ],
    template="""
You are an acrostic generating AI. I will give you some text and an acrostic phrase. Your job is to rewrite the text so that the first letter in each sentence spells out the acrostic phrase, while still preserving the meaning of the original text.

We'll do one sentence at a time. I'll give you the first letter of the new sentence, and your job will be to write a new sentence that starts with that letter.
I want you to make {n_attempts} unique attempts at writing the new sentence. Format your answer as a numbered list, with each attempt on a separate line. For example:
1. This is my first attempt.
2. This is my second attempt.
...
etc.

Then I'll check your work. If any of the sentences look good, I'll add it to the rewritten text and we'll move on to the next sentence. If not, I'll ask you to try again.

Original text: "{original_text}"
Acrostic phrase: "{acrostic_phrase}"
Rewritten text so far: "{rewritten_text}"
Starting letter for the next sentence: {current_starting_letter}

You will usually want to focus on a single sentence from the original text at a time and try to rewrite it. Here is the sentence you are currently working on:
"{current_original_sentence}"

Now provide your answer. Remember that the first word of your new sentence must begin with the letter {current_starting_letter} and that it should mean exactly the same thing as the corresponding sentence in the original text.
""",
)
