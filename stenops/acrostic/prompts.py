from langchain.prompts import PromptTemplate

original_text_summarizer_prompt = PromptTemplate(
    input_variables=["original_text"],
    template="""
I will give you some text and your job is write a bulleted list that clarifies the meaning of each sentence.

Write a bullet point that explains each sentence in very clear language. There should be one bullet point for each sentence in the original text.

Here is the text:
{original_text}

Provide only the bulleted list as your output. Do not include any other text in your output.
""",
)

original_text_contextualizer_prompt = PromptTemplate(
    input_variables=["original_text"],
    template="""
I will give you some text and your job is write a few succinct sentences with your best guess at the context of the text.

Here is the text:
{original_text}

What is the purpose of this text? What kind of person is the author and who might they be speaking to? What is the style of the text?
""",
)


rewritten_text_summarizer_prompt = PromptTemplate(
    input_variables=["rewritten_text"],
    template="""
You are a text summarizer. I will give you some text and your job is write a bulleted list summarizing the text.

If the there is no rewritten text yet, return just a space

Here is the text:
{rewritten_text}
""",
)


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
        "current_original_sentence",
        "original_text_summary",
        "original_text_context",
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

Here is your summary of the original text:
{original_text_summary}

Here is your reflection on the context and style of the original text:
{original_text_context}

Here's the sentence that AGA was attempting to rewrite:
"{current_original_sentence}"

Remember the following rules as you evaluate AGA's attempts:
- The first word of your new sentence must begin with the letter {current_starting_letter}.
- The new sentence should paraphrase the exact meaning of the corresponding sentence in the original text.
- The new sentence should be written with the same style and intent as the original text - assume the voice of the original author as you write.
- We want the rewritten text to sound natural, so try to avoid using words that are too obscure or unusual, and make sure that the sentences flow well together.
- Avoid changing the tense or construction of the original sentence - e.g. don't pretend to be answering a question if the original sentence was a statement.

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
        # "current_idea",
        "current_original_sentence",
        "original_text_summary",
        "original_text_context",
    ],
    template="""
You are an acrostic generating AI. I will give you some text and an acrostic phrase.
Your job is to rewrite the text so that the first letter in each sentence spells out the acrostic phrase, while still preserving the meaning and style of the original text.

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

Remember, it is crucial to preserve the meaning and style of the original text - so we'll start by analyzing that text.

Here is your summary of the original text:
{original_text_summary}

Here is your reflection on the context and style of the original text:
{original_text_context}

Make sure that your next sentence helps us build towards a perfect paraphrasing of the original text.

Here's the sentence you should try to rewrite for this attempt:
"{current_original_sentence}"

Remember the following rules as you work:
- The first word of your new sentence must begin with the letter {current_starting_letter}.
- The new sentence should paraphrase the exact meaning of the corresponding sentence in the original text.
- The new sentence should be written with the same style and intent as the original text - assume the voice of the original author as you write.
- It's possible that your answer will be very similar or even identical to the original sentence in some cases.

Now provide your answer:
""",
)
