# tests/test_explain.py

from unittest.mock import MagicMock, patch

# Import the function we want to test directly from its module
from statlingua.explain import explain

# A simple mock class to simulate a statsmodels OLSResults object
class MockOLSResults:
    def summary(self):
        return "--- MOCK OLS SUMMARY ---"

# Mock class for GLM results
class MockGLMResults:
    class MockFamily:
        __class__ = type("MockFamily", (), {"__name__": "Poisson"})
    
    def __init__(self):
        self.model = MagicMock()
        self.model.family = self.MockFamily()

    def summary(self):
        return "--- MOCK GLM SUMMARY ---"

# The patch target is the absolute path to the object we need to mock
# In this case, it's the `completion` function in the top-level `litellm` package
@patch('litellm.completion')
def test_explain_calls_litellm_with_correct_prompts(mock_completion: MagicMock):
    """
    Tests that explain() calls litellm.completion with correctly assembled prompts.
    """
    # --- Arrange ---
    # 1. Configure the mock to return a predictable response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a mock explanation."
    mock_completion.return_value = mock_response

    # 2. Create a mock model object
    mock_model = MockOLSResults()
    
    # --- Act ---
    # Call the function under test
    result = explain(
        model_object=mock_model,
        model="gpt-4o",
        context="A test context for the model.",
        audience="researcher"
    )

    # --- Assert ---
    # 1. Check that the mock was called exactly once
    mock_completion.assert_called_once()
    
    # 2. Get the keyword arguments passed to the mock
    call_kwargs = mock_completion.call_args.kwargs
    
    # 3. Check arguments with pytest's plain assert
    assert call_kwargs['model'] == "gpt-4o"
    
    # 4. Check the messages payload
    messages = call_kwargs['messages']
    system_prompt = messages[0]['content']
    user_prompt = messages[1]['content']
    
    assert "Target Audience: Researcher" in system_prompt
    assert "--- MOCK OLS SUMMARY ---" in user_prompt
    assert "A test context for the model." in user_prompt
    
    # 5. Check the function's return value
    assert result['text'] == "This is a mock explanation."

@patch('litellm.completion')
def test_explain_handles_glm(mock_completion: MagicMock):
    """
    Tests that the GLM handler is correctly used.
    """
    # Arrange
    mock_completion.return_value = MagicMock()
    mock_glm = MockGLMResults()

    # Act
    explain(model_object=mock_glm, model="gpt-4o")

    # Assert
    mock_completion.assert_called_once()
    user_prompt = mock_completion.call_args.kwargs['messages'][1]['content']
    
    # Check that the user prompt identifies the model as a "glm"
    assert "Explain the following glm model output:" in user_prompt
    assert "--- MOCK GLM SUMMARY ---" in user_prompt
