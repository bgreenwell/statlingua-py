# src/statlingua/prompts.py

import importlib.resources
from pathlib import Path

def _read_prompt_file(path_parts: list[str]) -> str:
    """Reads a prompt file from the package's data.

    Parameters
    ----------
    path_parts : list[str]
        A list of path components (directories and filename).

    Returns
    -------
    str
        The content of the prompt file, or an empty string if not found.
    """
    try:
        # This is the modern way to access package data files
        return importlib.resources.files('statlingua') \
            .joinpath('prompts', *path_parts).read_text(encoding='utf-8')
    except (FileNotFoundError, NotADirectoryError):
        # Gracefully handle cases where a prompt file might be missing
        return ""

def assemble_sys_prompt(model_name: str, audience: str, verbosity: str, style: str) -> str:
    """Assembles the complete system prompt from various markdown files.

    This function dynamically constructs the system prompt sent to the LLM
    by combining base roles, audience-specific instructions, verbosity levels,
    and model-specific guidelines.

    Parameters
    ----------
    model_name : str
        The internal name of the model (e.g., "lm", "glm"), which
        corresponds to a directory in `prompts/models/`.
    audience : str
        The target audience (e.g., "novice", "researcher").
    verbosity : str
        The desired level of detail (e.g., "brief", "detailed").
    style : str
        The desired output format (e.g., "markdown", "json").

    Returns
    -------
    str
        The fully constructed system prompt.
    """
    # Base role and model-specific role
    role_base = _read_prompt_file(["common", "role_base.md"])
    role_specific = _read_prompt_file(["models", model_name, "role_specific.md"])
    role_section = f"## Role\n\n{role_base}\n\n{role_specific}".strip()

    # Audience and verbosity
    audience_text = _read_prompt_file(["audience", f"{audience}.md"])
    verbosity_text = _read_prompt_file(["verbosity", f"{verbosity}.md"])
    audience_section = (
        f"## Intended Audience and Verbosity\n\n"
        f"### Target Audience: {audience.title()}\n{audience_text}\n\n"
        f"### Level of Detail (Verbosity): {verbosity.title()}\n{verbosity_text}"
    ).strip()

    # Response format
    style_text = _read_prompt_file(["style", f"{style}.md"])
    style_section = (
        f"## Response Format Specification (Style: {style.title()})\n\n{style_text}"
    ).strip()

    # Model-specific instructions
    instructions_text = _read_prompt_file(["models", model_name, "instructions.md"])
    # Fallback to default instructions if model-specific ones don't exist
    if not instructions_text.strip():
        instructions_text = _read_prompt_file(["models", "default", "instructions.md"])
    instructions_section = f"## Instructions\n\n{instructions_text}".strip()

    # Final caution
    caution_text = _read_prompt_file(["common", "caution.md"])
    caution_section = f"## Caution\n\n{caution_text}".strip()

    # Assemble all sections into the final prompt
    full_prompt = "\n\n\n".join([
        role_section,
        audience_section,
        style_section,
        instructions_section,
        caution_section
    ])
    return full_prompt.strip()

def build_user_prompt(model_description: str, output: str, context: str = None) -> str:
    """Builds the user prompt containing the model summary.

    Parameters
    ----------
    model_description : str
        A brief description of the model type.
    output : str
        The captured summary output from the statistical model.
    context : str, optional
        Additional user-provided context about the data or research question.

    Returns
    -------
    str
        The fully constructed user prompt.
    """
    prompt = f"Explain the following {model_description} output:\n\n---\n\n{output}"
    if context and context.strip():
        prompt += f"\n\n---\n\n## Additional context to consider\n\n{context.strip()}"
    return prompt
