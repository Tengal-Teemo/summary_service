import difflib
import anthropic
import dotenv
import os
import re

import numpy as np

import tiktoken

from pprint import pprint
from wordllama import WordLlama
from wordllama.wordllama import WordLlamaInference

from text_extractor.content_handler import file_to_plaintext
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


model = None

def load_embedding_model() -> WordLlamaInference:
    # return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return WordLlama.load()


def embed(paragraph: str) -> np.ndarray:
    global model
    if model is None:
        model = load_embedding_model()
    return model.embed(paragraph)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    if vec1 is None or vec2 is None:
        return 0
    return abs(np.dot(vec1, vec2.T) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

enc = tiktoken.get_encoding("o200k_base")

def query_haiku(prompt: str, system_prompt: str, temperature: float = 0.7):
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=temperature,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text


def persona_from_role(role: str) -> str:
    prompt = """A persona is a concise paragraph of text meant to encapsulate a person.  The purpose of this is that a persona paragraph can be embedded to allow for cosine similarity comparison of various characters, in order to determine relevance.  A persona shouldn't contain personal details like names, however they can contain characteristic adjectives.
    Please create a persona for a <role>{role}</role>. Wrap it in <persona></persona> tags, and please be concise and begin the persona with the provided <role></role>.""".format(role=role)
    system_prompt = f"You are Claude, an AI assistant, created to help summarize changes in documents in a helpful, harmless manner."
    persona_response = query_haiku(prompt, system_prompt, temperature=0.7)
    persona = re.search(r'<persona>((\n|.)*)</persona>', persona_response).groups()[0]
    return persona


def summary_from_files(fv1p: str, fv2p: str, role_personas: dict[str, str]) -> (list[dict], list[list[int]], int, int):
    """

    :param fv1p: path to v1 of the file
    :param fv2p: path to v2 of the file
    :param role_personas: list of personas to rank the relevant changes by
    :return:
        summarized diffs dict {'summary', 'persona', 'persona_embedding'}
        persona map list, [[p1 most relevant -> p1 least relevant]...]
        prompt tokens int
        response tokens int
    """
    if fv1p.split('.')[-1] != fv2p.split('.')[-1]:
        raise ValueError('file extensions don\'t match')

    fv1_text = file_to_plaintext(fv1p)[0].split('\n')
    fv2_text = file_to_plaintext(fv2p)[0].split('\n')

    # TODO: expand tokens around diffs if below certain threshold
    f_unified_diff = difflib.unified_diff(fv1_text, fv2_text, n=2)
    f_unified_diffs = list(f_unified_diff)

    # Generate diffs
    diff_texts = []
    for diff in f_unified_diffs:
        if diff.startswith('@@') and diff.endswith('@@\n'):
            diff_texts.append(diff)
        elif diff_texts:
            diff_texts[-1] += '\n'+diff

    summarized_diffs = []

    prompt_tokens = 0
    response_tokens = 0

    system_prompt = f"You are Claude, an AI assistant, created to help summarize changes in documents in a helpful, harmless manner."
    for diff in diff_texts:

        prompt = (f'Summarize changes in the unified diff present in the <edit></edit> tags.  Wrap the changes in some <summary></summary> tags.\n\n'
                  f'<edit>\n{diff}</edit>')

        summary_response = query_haiku(prompt, system_prompt)

        prompt_tokens += len(enc.encode(system_prompt)) + len(enc.encode(prompt))
        response_tokens += len(enc.encode(summary_response))

        summary = re.search(r'<summary>((.|\n)*)</summary>', summary_response).groups()[0]

        prompt = (f'A persona is a concise description of a role someone might have, such as a CEO, Historian, Student.\n\n'
                  f'Given the text below in the <text></text> tags, create a persona representing who would be interested in this text, devoid of personal details like names, describing the kind of person who would be interested.  Do this concisely within <persona></persona> tags\n\n'
                  f'<text>\n'
                  f'{summary}'
                  f'\n</text>')
        persona_response = query_haiku(prompt, system_prompt)

        # counting the tokens
        prompt_tokens += len(enc.encode(system_prompt)) + len(enc.encode(prompt))
        response_tokens += len(enc.encode(persona_response))

        persona = re.search(r'<persona>((.|\n)*)</persona>', persona_response).groups()[0]

        summarized_diffs.append({'summary': summary, 'persona': persona})

    persona_labels = list(role_personas.keys())
    personas = list(role_personas.values())
    persona_embeddings = [embed(role_persona) for role_persona in role_personas.values()]

    for count, summarized_diff in enumerate(summarized_diffs):
        summarized_diffs[count]['persona_embedding'] = embed(summarized_diff['persona'])

    # with persona_embedding information, we now create an order of relevance per
    persona_scores = []
    for persona, embedding in zip(role_personas, persona_embeddings):
        # this map will be sorted based on cosine similarity of persona embedding and index
        scores = [cosine_similarity(summarized_diff['persona_embedding'], embedding) for summarized_diff in summarized_diffs]
        persona_scores.append(scores)

    # combine relevant paragraphs
    prompt_template = """Take the information below in the <change></change> tags, and summarize them in a concise, bullet-point list.\n
    The summary should be contained in <summary></summary> tags.\n\n{changes}"""
    threshold = 0.5
    persona_summaries = {}
    for persona_role, persona, persona_score in zip(role_personas, personas, persona_scores):
        score_indices = list(range(len(persona_score)))
        prompt_change = "\n\n".join([f'<change>\n{summarized_diffs[index]["summary"]}\n</change>' for index, score in zip(score_indices, persona_score) if float(score) > threshold])
        if prompt_change:
            prompt = prompt_template.format(changes=prompt_change)
            prompt_tokens += len(enc.encode(system_prompt)) + len(enc.encode(prompt))
            summary_response = query_haiku(prompt, system_prompt)
            response_tokens += len(enc.encode(summary_response))
            response_summary = re.search(r'<summary>((.|\n)*)</summary>', summary_response).groups()[0]
        else:
            response_summary = ""
        persona_summaries.update({persona_role: response_summary})

    return persona_summaries, prompt_tokens, response_tokens
