"""Backend for All OpenAI Compatable API"""

import json
import logging
import time
import os
from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
# from pprint import pprint

logger = logging.getLogger(__name__)

_client: openai.OpenAI = None  # type: ignore
# api_key = os.getenv("API_KEY") # set openai api key
# base_url = os.getenv("BASE_URL")
api_key = os.environ['API_KEY']

OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)
gpt4o_base_url= {"base_url": "https://api.openai.com/v1"}


def _setup_openai_client(base_url, api_key=api_key):
    global _client
    _client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

def clean_json_response(response_text: str) -> str:
    # 移除 markdown 的 ```json 和 ``` 标记
    response_text = response_text.strip()
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    elif response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    return response_text.strip()


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    """
    Query the OpenAI API, optionally with function calling.
    If the model doesn't support function calling, gracefully degrade to text generation.
    """
    current_base_url = gpt4o_base_url["base_url"] if "gpt-4o" in model_kwargs["model"] else os.environ.get('BASE_URL')
    _setup_openai_client(api_key=api_key, base_url=current_base_url)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)
    
    if "claude-" in model_kwargs["model"]:
        # Anthropic doesn't allow not having user messages
        # if we only have system msg -> use it as user msg
        if system_message is not None and user_message is None:
            system_message, user_message = user_message, system_message

        # Anthropic passes system messages as a separate argument
        if system_message is not None:
            filtered_kwargs["system"] = system_message

    # Convert system/user messages to the format required by the client
    messages = opt_messages_to_list(system_message, user_message)

    # If the model is a Gemini model and function specification is provided,
    if "gemini-" in model_kwargs["model"] and func_spec is not None:
        # Get the function specification for the tool
        tool_spec = func_spec.as_openai_tool_dict["function"]
        
        # Set the function call to the tool specification
        function_prompt = f"""
Please provide your response in the following JSON format that matches this function specification:
Function Name: {tool_spec["name"]}
Description: {tool_spec["description"]}
Parameters: {json.dumps(tool_spec["parameters"], indent=2)}

Your response should be a valid JSON object with these parameters.
"""
        # Add the function call to the messages
        if user_message:
            messages.append({
                "role": "user",
                "content": user_message + "\n\n" + function_prompt
            })
        else:
            messages.append({
                "role": "user",
                "content": function_prompt
            })
    else:
        # For other models, if function specification is provided,
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    completion = None
    t0 = time.time()

    # Attempt the API call
    try:
        completion = backoff_create(
            _client.chat.completions.create,
            OPENAI_TIMEOUT_EXCEPTIONS,
            messages=messages,
            **filtered_kwargs,
        )
    except openai.BadRequestError as e:
        
        if "gemini-" not in model_kwargs["model"]:
            if "function calling" in str(e).lower() or "tools" in str(e).lower():
                logger.warning(
                    "Function calling was attempted but is not supported by this model. "
                    "Falling back to plain text generation."
                )
                filtered_kwargs.pop("tools", None)
                filtered_kwargs.pop("tool_choice", None)

                completion = backoff_create(
                    _client.chat.completions.create,
                    OPENAI_TIMEOUT_EXCEPTIONS,
                    messages=messages,
                    **filtered_kwargs,
                )
            else:
                raise
        else:
            raise

    req_time = time.time() - t0
    choice = completion.choices[0]

    # deal with function calling
    if "gemini-" in model_kwargs["model"] and func_spec is not None:
        try:
            cleaned_response = clean_json_response(choice.message.content)
            output = json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse Gemini response as JSON: {str(e)}\n"
                f"Original response: {choice.message.content}\n"
                f"Cleaned response: {cleaned_response}"
            )
            output = choice.message.content
    else:
        if func_spec is None or "tools" not in filtered_kwargs:
            output = choice.message.content
        else:
            tool_calls = getattr(choice.message, "tool_calls", None)
            if not tool_calls:
                output = choice.message.content
            else:
                first_call = tool_calls[0]
                if first_call.function.name != func_spec.name:
                    output = choice.message.content
                else:
                    try:
                        output = json.loads(first_call.function.arguments)
                    except json.JSONDecodeError as ex:
                        logger.error(
                            "Error decoding function arguments:\n"
                            f"{first_call.function.arguments}"
                        )
                        raise ex

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
