"""
Test suite for the ANN System.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from lang_graph.ann_system import (
    AnnWorkflow, 
    AgentState, 
    LLMClient, 
    CodeExecutor,
    planner_node_logic,
    executor_node_logic,
    critique_node_logic
)

class TestLLMClient:
    """Test LLM client functionality."""
    
    def test_initialization_with_openai_key(self):
        """Test client initialization with OpenAI API key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            with patch('openai.OpenAI'):
                client = LLMClient()
                assert client.openai_client is not None
                assert client.gemini_client is None
    
    def test_initialization_with_gemini_key(self):
        """Test client initialization with Gemini API key."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    client = LLMClient()
                    assert client.openai_client is None
                    assert client.gemini_client is not None
    
    def test_call_llm_openai(self):
        """Test OpenAI LLM call."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            mock_openai = Mock()
            mock_response = Mock()
            mock_response.choices[0].message.content = "Test response"
            mock_openai.chat.completions.create.return_value = mock_response
            
            with patch('openai.OpenAI', return_value=mock_openai):
                client = LLMClient()
                result = client.call_llm("Test prompt", "gpt-5-mini")
                assert result == "Test response"
    
    def test_call_llm_gemini(self):
        """Test Gemini LLM call."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            mock_genai = Mock()
            mock_response = Mock()
            mock_response.text = "Test response"
            mock_genai.generate_content.return_value = mock_response
            
            with patch('google.generativeai.GenerativeModel', return_value=mock_genai):
                with patch('google.generativeai.configure'):
                    client = LLMClient()
                    result = client.call_llm("Test prompt", "gemini-1.5-flash")
                    assert result == "Test response"

class TestCodeExecutor:
    """Test code execution functionality."""
    
    def test_validate_python_code_valid(self):
        """Test validation of valid Python code."""
        executor = CodeExecutor()
        valid_code = "print('Hello, World!')"
        assert executor._validate_python_code(valid_code) is True
    
    def test_validate_python_code_invalid_syntax(self):
        """Test validation of invalid Python syntax."""
        executor = CodeExecutor()
        invalid_code = "print('Hello, World!'"
        assert executor._validate_python_code(invalid_code) is False
    
    def test_validate_python_code_dangerous_operations(self):
        """Test validation of dangerous operations."""
        executor = CodeExecutor()
        dangerous_code = "import os\nos.system('rm -rf /')"
        assert executor._validate_python_code(dangerous_code) is False
    
    def test_extract_code_from_markdown(self):
        """Test code extraction from markdown."""
        executor = CodeExecutor()
        markdown_content = "```python\nprint('Hello')\n```"
        extracted = executor._extract_code_from_markdown(markdown_content)
        assert extracted == "print('Hello')"
    
    def test_extract_code_no_markdown(self):
        """Test code extraction without markdown."""
        executor = CodeExecutor()
        plain_code = "print('Hello')"
        extracted = executor._extract_code_from_markdown(plain_code)
        assert extracted == "print('Hello')"

class TestAnnWorkflow:
    """Test workflow functionality."""
    
    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = AnnWorkflow(max_revisions=3)
        assert workflow.max_revisions == 3
        assert workflow.graph is not None
    
    def test_should_continue_completion(self):
        """Test workflow continuation logic for completion."""
        workflow = AnnWorkflow()
        state = {"critique": "DONE", "revision_number": 1}
        result = workflow.should_continue(state)
        assert result == "end"
    
    def test_should_continue_revision_limit(self):
        """Test workflow continuation logic for revision limit."""
        workflow = AnnWorkflow(max_revisions=3)
        state = {"revision_number": 3, "critique": "Need improvement"}
        result = workflow.should_continue(state)
        assert result == "end"
    
    def test_should_continue_continue(self):
        """Test workflow continuation logic for continuing."""
        workflow = AnnWorkflow()
        state = {"revision_number": 1, "critique": "Need improvement"}
        result = workflow.should_continue(state)
        assert result == "continue"
    
    def test_should_continue_error(self):
        """Test workflow continuation logic for errors."""
        workflow = AnnWorkflow()
        state = {"error": "Something went wrong"}
        result = workflow.should_continue(state)
        assert result == "error"

class TestNodes:
    """Test node logic functionality."""
    
    @patch('lang_graph.ann_system.nodes.call_llm')
    def test_planner_node_logic_success(self, mock_call_llm):
        """Test successful planning node execution."""
        mock_call_llm.return_value = "Test plan"
        state = {"initial_task": "Test task", "history": []}
        
        result = planner_node_logic(state)
        
        assert result["plan"] == "Test plan"
        mock_call_llm.assert_called_once()
    
    @patch('lang_graph.ann_system.nodes.call_llm')
    def test_planner_node_logic_empty_response(self, mock_call_llm):
        """Test planning node with empty response."""
        mock_call_llm.return_value = ""
        state = {"initial_task": "Test task", "history": []}
        
        with pytest.raises(RuntimeError, match="Planner returned empty response"):
            planner_node_logic(state)
    
    @patch('lang_graph.ann_system.nodes.call_llm')
    @patch('lang_graph.ann_system.nodes.execute_python_code')
    def test_executor_node_logic_success(self, mock_execute, mock_call_llm):
        """Test successful executor node execution."""
        mock_call_llm.return_value = "```python\nprint('Hello')\n```"
        mock_execute.return_value = "Execution Result: SUCCESS"
        state = {"plan": "Test plan"}
        
        result = executor_node_logic(state)
        
        assert result["code"] == "```python\nprint('Hello')\n```"
        assert result["execution_result"] == "Execution Result: SUCCESS"
    
    def test_executor_node_logic_no_plan(self):
        """Test executor node without plan."""
        state = {}
        
        with pytest.raises(RuntimeError, match="No plan available for execution"):
            executor_node_logic(state)
    
    @patch('lang_graph.ann_system.nodes.call_llm')
    def test_critique_node_logic_success(self, mock_call_llm):
        """Test successful critique node execution."""
        mock_call_llm.return_value = "DONE"
        state = {
            "plan": "Test plan",
            "code": "Test code",
            "execution_result": "Success"
        }
        
        result = critique_node_logic(state)
        
        assert result["critique"] == "DONE"
    
    def test_critique_node_logic_missing_state(self):
        """Test critique node with missing state."""
        state = {"plan": "Test plan"}
        
        with pytest.raises(RuntimeError, match="Missing required state for critique"):
            critique_node_logic(state)

if __name__ == "__main__":
    pytest.main([__file__])
