import re
import ast

def is_legal_text(text):
    if re.search(r'[^\x00-\x7F]',  text):
        return False
    try:
        ast.literal_eval(text)
        return True
    except:
        return False
    
def abstract_text(text, target):
    matches = list(
        map(
            lambda s: any(map(lambda t: t in s, target)),
            text
        )
    )
    return list(
        map(
            lambda _: any(_),
            zip(
                matches, [False] + matches[:-1], matches[1:] + [False]
            )
        )
    )

from openai import OpenAI

API_KEY = 'sk-7e6e1a9c2ba84b18a09372aff22ff837' # 已失效
API_MODEL = 'deepseek-chat'
API_URL = 'https://api.deepseek.com/v1'

client = OpenAI(
    api_key=API_KEY,
    base_url=API_URL,
)

def ask_deepseek(ask):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides accurate and concise answers to user queries."
        },
        {
            "role": "user",
            "content": ask
        }
    ]
    return client.chat.completions.create(
        model=API_MODEL,
        messages=messages,
        stream=False
    ).choices[0].message.content

def get_pos_weights(df, targets):
    f = lambda x: x[0] / x[1]
    weights = {
        key: f(df[key].value_counts()[[0, 1]].to_numpy())
        for key in targets
    }
    return weights
