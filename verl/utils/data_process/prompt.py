"""
Prompt utils for data preprocessing
"""

ZERO_STYLE_PROMPT_TEMPLATE = """{{START_TOKEN}}{{system_symbol}}
{{system_prompt}}{{END_TOKEN}}
{{START_TOKEN}}{{user_symbol}}
{{prompt}}{{extra_instruction}}{{END_TOKEN}}
{{START_TOKEN}}{{assistant_symbol}}
<think>"""

# deprecated
ZERO_STYLE_PROMPT_TEMPLATE_2 = """{{START_TOKEN}}{{user_symbol}}
{{prompt}}{{extra_instruction}}{{END_TOKEN}}
{{START_TOKEN}}{{assistant_symbol}}
<think>"""

SYSTEM_PROMPT = """A conversation between a user and an assistant. The assistant first thinks through the problem step-by-step inside <think>...</think>, then provides the final response to user."""
SYSTEM_SYMBOL = "system"
USER_SYMBOL = "user"
ASSISTANT_SYMBOL = "assistant"
START_TOKEN = "<|im_start|>"
END_TOKEN = "<|im_end|>"


def build_zero_style_prompt(
    template: str = ZERO_STYLE_PROMPT_TEMPLATE, 
    prompt: str = "",
    extra_instruction: str = "",
    model_name: str = "Qwen/Qwen2.5-7B"
    ) -> str:
    if "Qwen" in model_name:
        replacements = {
            "{{START_TOKEN}}": START_TOKEN,
            "{{END_TOKEN}}": END_TOKEN,
            "{{system_symbol}}": SYSTEM_SYMBOL,
            "{{user_symbol}}": USER_SYMBOL,
            "{{assistant_symbol}}": ASSISTANT_SYMBOL,
            "{{system_prompt}}": SYSTEM_PROMPT,
            "{{prompt}}": prompt,
            "{{extra_instruction}}": extra_instruction,
        }

        for key, val in replacements.items():
            template = template.replace(key, val)

        if extra_instruction:
            template = template.replace("{{extra_instruction}}", " " + extra_instruction)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only Qwen is supported for now.")
    
    return template


if __name__ == "__main__":
    print("=" * 10)
    prompt = "What is the sum of 1 and 2?"
    print(build_zero_style_prompt(template=ZERO_STYLE_PROMPT_TEMPLATE, prompt=prompt))

    print("=" * 10)
    prompt = "First thinks through the problem step-by-step inside <think>...</think>, then provides the final answer. What is the sum of 1 and 2?"
    print(build_zero_style_prompt(template=ZERO_STYLE_PROMPT_TEMPLATE_2, prompt=prompt))
