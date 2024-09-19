import difflib
import anthropic
import dotenv
import os
import re

import numpy as np

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


def query_haiku(prompt: str, system_prompt: str):
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        temperature=0.5,
        system=system_prompt,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content


def summary_from_files(fv1p: str, fv2p: str, role_persona: str) -> list[dict]:

    if fv1p.split('.')[-1] != fv2p.split('.')[-1]:
        raise ValueError('file extensions don\'t match')

    fv1_text = file_to_plaintext(fv1p)[0].split('\n')
    fv2_text = file_to_plaintext(fv2p)[0].split('\n')

    f_unified_diff = difflib.unified_diff(fv1_text, fv2_text)
    f_unified_diffs = list(f_unified_diff)

    # Generate diffs
    diff_texts = []
    for diff in f_unified_diffs:
        if diff.startswith('@@') and diff.endswith('@@\n'):
            diff_texts.append(diff)
        elif diff_texts:
            diff_texts[-1] += '\n'+diff

    summarized_diffs = []

    for diff in diff_texts:
        system_prompt = f"You are Claude, an AI assistant, created to help summarize changes in documents in a helpful, harmless manner."
        prompt = (f'Summarize changes in the unified diff present in the <edit></edit> tags.  Wrap the changes in some <summary></summary> tags.\n\n'
                  f'<edit>\n{diff}</edit>')
        summary_response = query_haiku(prompt, system_prompt)[0].text

        summary = re.match(r'<summary>((.|\n)*)</summary>', summary_response).groups()[0]

        prompt = (f'Given the following text, in the given <text></text> tags, who is most likely to find it important?'
                  f'Describe the persona concisely within <affects></affects> tags\n\n'
                  f'<text>\n'
                  f'{summary}'
                  f'\n</text>')
        persona_response = query_haiku(prompt, system_prompt)[0].text

        persona = re.findall(r'<affects>(.*)</affects>', persona_response)

        summarized_diffs.append({'summary': summary, 'persona': persona})

    model = load_embedding_model()
    persona_embedding = model.embed(role_persona)

    for count, summarized_diff in enumerate(summarized_diffs):
        affects_similarities = cosine_similarity(persona_embedding, model.embed(summarized_diff['persona']))
        summarized_diffs[count]['relevance'] = affects_similarities

    summarized_diffs.sort(key=lambda x: x['relevance'], reverse=True)

    return summarized_diffs