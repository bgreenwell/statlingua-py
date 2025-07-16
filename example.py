import os
import pandas as pd
import statsmodels.api as sm
from statlingua import explain

def run_example():
    """
    Runs a complete example of fitting a model and explaining it with statlingua.
    """
    # --- 1. Set up your API Key ---
    # IMPORTANT: Replace with your actual API key or set it as an environment variable.
    # For example: export OPENAI_API_KEY="sk-..."
    # litellm will automatically find keys set as environment variables.
    # If you need to set it in the script, uncomment the following line:
    # os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY environment variable.")
        return

    # --- 2. Create a simple dataset ---
    # Using a classic dataset about car speed and stopping distances.
    data = {
        'speed': [4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25],
        'dist': [2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46, 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46, 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85]
    }
    df = pd.DataFrame(data)

    # For statsmodels, we need to manually add a constant (intercept)
    X = sm.add_constant(df['speed'])
    y = df['dist']

    # --- 3. Fit a statistical model ---
    # We'll fit a simple linear regression model.
    print("Fitting linear model...")
    model = sm.OLS(y, X).fit()
    print("Model fitting complete.")
    # print(model.summary()) # You can uncomment this to see the raw output

    # --- 4. Get an explanation from statlingua ---
    # We provide context to help the LLM give a better explanation.
    model_context = (
        "This model analyzes the 'cars' dataset from the 1920s. "
        "The goal is to understand how a car's speed (in mph) affects its "
        "stopping distance (in feet)."
    )
    
    print("\nCalling statlingua.explain() to get an explanation...")
    explanation = explain(
        model_object=model,
        model="gpt-4o",  # Specify the LLM you want to use
        context=model_context,
        audience="student",
        verbosity="detailed"
    )
    print("Explanation received.")

    # --- 5. Print the result ---
    print("\n--- Statlingua Explanation ---")
    print(explanation['text'])
    print("----------------------------")


if __name__ == '__main__':
    run_example()

