# statlingua

[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

An experimental python package to help you understand and diagnose statistical models using the power of large language models.

> **Note:** This package is under active development. The API is subject to change, and users should expect rapid evolution of features.

This package provides tools to translate the complex output of statistical models into clear, human-readable explanations and to help diagnose model assumptions through an interactive, agent-based workflow. It is designed for students, researchers, and data scientists who want to gain a deeper intuition for their models.

## Core features

  * **Explain model results:** Get detailed, context-aware explanations of your model's summary output using the `explain()` function.
  * **Diagnose model assumptions:** Use the agentic `diagnose_agent()` function to interactively check key statistical assumptions like linearity, homoscedasticity, and more.
  * **Powered by modern LLMs:** Leverages the vision and reasoning capabilities of models like GPT-4o and Google's Gemini to interpret diagnostic plots and provide insightful advice.
  * **Extensible by design:** Built to be easily extended with support for new statistical models and diagnostic tools.

## Installation

You can install the package directly from GitHub using `pip`.

```sh
pip install git+https://github.com/your-username/statlingua-py.git
```

You will also need to install the dependencies required for running statistical models and plotting:

```sh
pip install statsmodels matplotlib seaborn pandas
```

## Quick start

Here is a simple example of how to use `statlingua` to explain a model and then diagnose its assumptions. Before running, make sure you have set your `OPENAI_API_KEY` (or the key for your preferred provider) as an environment variable.

```python
import os
import statsmodels.api as sm
from statlingua import explain, diagnose_agent

# Ensure your API key is set
# export OPENAI_API_KEY="sk-..."

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set your OPENAI_API_KEY environment variable.")

# 1. Load data and fit a model
duncan_data = sm.datasets.duncan.load()
y = duncan_data.data['prestige']
X = duncan_data.data[['income', 'education']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 2. Get a high-level explanation of the model results
print("--- Getting Model Explanation ---")
explanation = explain(
    model_object=model,
    model="gpt-4o",
    audience="student"
)
print(explanation['text'])
print("-" * 30)

# 3. Start a diagnostic session with the agent
print("\n--- Starting Diagnostic Session ---")
user_question = (
    "Can you check if the relationship in my model is linear and if the "
    "variance of the residuals is constant?"
)

diagnostic_result = diagnose_agent(
    model_object=model,
    prompt=user_question,
    model="gpt-4o"  # Use a vision-capable model
)

print(diagnostic_result['text'])
if diagnostic_result.get('plot'):
    print(f"\nA plot was generated at: {diagnostic_result['plot']}")

```

## Contributing

Contributions are welcome\! If you have suggestions for new features, find a bug, or want to add support for a new model, please open an issue on the GitHub repository.

## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3).