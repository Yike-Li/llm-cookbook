
from openai import OpenAI
import hvac

vault_client = hvac.Client()
if not vault_client.is_authenticated():
    raise ValueError(
        "An OpenAI API key wasn't found in the configs or environment and vault isn't authenticated, giving up."
    )
response = vault_client.read("secret-paas/project/octo/dev/env/OPENAI_API_KEY")

if isinstance(response, dict):
    api_key = response["data"]["value"]
else:
    raise ValueError(
        "Received unexpected response from vault while fetching OpenAI key."
    )
print(api_key)

# api_key = os.environ.get("PERPLEXITY_API_KEY")

client = OpenAI(api_key=api_key)

# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果


def get_completion(prompt, model="gpt-4o-mini", temperature=0):
    '''
    prompt: 对应的提示词
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4

    '''

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # 模型输出的温度系数，控制输出的随机程度
    )
    # 调用 OpenAI 的 ChatCompletion 接口
    return response.choices[0].message.content

    # stream = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     stream=True,
    # )
    # response = []
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         response.append(chunk.choices[0].delta.content)

    # return ('').join(response)


def get_completion_from_messages(messages, model="gpt-4o-mini", temperature=0, max_tokens=None):
    '''
    封装一个支持更多参数的自定义访问 OpenAI GPT3.5 的函数

    参数: 
    messages: 这是一个消息列表，每个消息都是一个字典，包含 role(角色）和 content(内容)。角色可以是'system'、'user' 或 'assistant’，内容是角色的消息。
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)，有内测资格的用户可以选择 gpt-4
    temperature: 这决定模型输出的随机程度，默认为0，表示输出将非常确定。增加温度会使输出更随机。
    max_tokens: 这决定模型输出的最大的 token 数。
    '''
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,  # 控制模型输出的随机程度
        max_tokens=max_tokens,  # 控制输出的最大长度
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message.content


def get_completion_and_token_count(messages,
                                   model="gpt-3.5-turbo",
                                   temperature=0,
                                   max_tokens=None):
    """
    使用 OpenAI 的 GPT-3 模型生成聊天回复，并返回生成的回复内容以及使用的 token 数量。

    参数:
    messages: 聊天消息列表。
    model: 使用的模型名称。默认为"gpt-3.5-turbo"。
    temperature: 控制生成回复的随机性。值越大，生成的回复越随机。默认为 0。
    max_tokens: 生成回复的最大 token 数量。默认为 500。

    返回:
    content: 生成的回复内容。
    token_dict: 包含'prompt_tokens'、'completion_tokens'和'total_tokens'的字典，分别表示提示的 token 数量、生成的回复的 token 数量和总的 token 数量。
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content

    token_dict = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens,
    }

    return content, token_dict


def get_moderation_output(message,
                          model="gpt-3.5-turbo",
                          temperature=0,
                          max_tokens=None):
    """
    使用 OpenAI 的 GPT-3 模型生成聊天回复，并返回生成的回复内容以及使用的 token 数量。

    参数:
    messages: 聊天消息列表。
    model: 使用的模型名称。默认为"gpt-3.5-turbo"。
    temperature: 控制生成回复的随机性。值越大，生成的回复越随机。默认为 0。
    max_tokens: 生成回复的最大 token 数量。默认为 500。

    返回:
    content: 生成的回复内容。
    token_dict: 包含'prompt_tokens'、'completion_tokens'和'total_tokens'的字典，分别表示提示的 token 数量、生成的回复的 token 数量和总的 token 数量。
    """
    response = client.moderations.create(
        input=message
    )
    moderation_output = response.results[0]

    return moderation_output



