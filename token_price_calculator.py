import tiktoken
import json

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def num_tokens_from_file(file_dir, model_name='gpt-3.5-turbo') -> float:
    """Returns the number of tokens in a jsonl file."""

    def num_tokens_from_string(string: str, model_name="gpt-3.5-turbo") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    #import jsonl file contents by list
    with open(file_dir, 'r') as file:
        sum_tokens = 0
        for line in file:
            file_content = json.loads(line)
            messages = file_content['messages']
            for m in range(len(messages)):
                string = messages[m]['content']
                # print(string)
                n_tokens = num_tokens_from_string(string, model_name)
                sum_tokens += n_tokens
                sum_tokens += 4  #message/role boundary token

    return sum_tokens
# train[0]

# tkn = num_tokens_from_file('./results/gpt-3.5-turbo_ko_243_train.jsonl')
# tkn2 = num_tokens_from_file('./results/gpt-3.5-turbo_ko_243_valid.jsonl')
# tkn * 0.0080 / 1000 + tkn2 * 0.0080 / 1000
