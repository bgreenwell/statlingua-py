# src/statlingua/explain.py

from typing import Any
import litellm

# Import our internal modules
from .prompts import assemble_sys_prompt, build_user_prompt
from .model_handlers import get_handler

def explain(
    model_object: Any,
    model: str,
    context: str = None,
    audience: str = "novice",
    verbosity: str = "moderate",
    style: str = "markdown",
    **kwargs: Any,
) -> dict:
    """Explains a statistical model's output using an LLM.

    This is the main function of the statlingua package. It takes a fitted
    statistical model object, captures its summary, and uses an LLM to
    generate a plain-language explanation tailored to a specific audience.

    Parameters
    ----------
    model_object : Any
        A fitted statistical model object from a supported library
        (e.g., a results object from statsmodels).
    model : str
        The model string for the LLM provider (e.g., "gpt-4o",
        "claude-3-opus-20240229"). This is passed directly to litellm.
    context : str, optional
        Additional context about the data or research question to provide
        to the LLM, by default None.
    audience : str, optional
        The target audience for the explanation. Must be one of "novice",
        "student", "researcher", "manager", or "domain_expert".
        Defaults to "novice".
    verbosity : str, optional
        The desired level of detail. Must be one of "brief", "moderate",
        or "detailed". Defaults to "moderate".
    style : str, optional
        The output format style. Must be one of "markdown", "html", "json",
        "text", or "latex". Defaults to "markdown".
    **kwargs : Any
        Additional keyword arguments to pass directly to the
        `litellm.completion` function. This can be used for parameters
        like `api_key`, `base_url` (which maps to `api_base`),
        `temperature`, `max_tokens`, etc.

    Returns
    -------
    dict
        A dictionary containing the explanation and metadata, with keys:
        'text', 'model_type', 'audience', 'verbosity', 'style'.
    """
    # 1. Get the model's summary and internal type name using the handler
    handler = get_handler(model_object)
    model_name, summary_text = handler(model_object)

    # 2. Assemble the system and user prompts
    system_prompt = assemble_sys_prompt(model_name, audience, verbosity, style)
    user_prompt = build_user_prompt(
        model_description=f"{model_name} model",
        output=summary_text,
        context=context
    )

    # Map common alias `base_url` to litellm's `api_base` if present
    if "base_url" in kwargs:
        kwargs["api_base"] = kwargs.pop("base_url")

    # 3. Call the LLM via litellm, passing all relevant parameters
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )

    # 4. Structure and return the output
    explanation_text = response.choices[0].message.content
    
    # The R version has a .remove_fences() utility. We can add this later
    # for now, we'll return the raw text.

    output = {
        "text": explanation_text,
        "model_type": model_name,
        "audience": audience,
        "verbosity": verbosity,
        "style": style,
    }
    return output

