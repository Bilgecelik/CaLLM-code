"""
Centralized repository for all prompts used in the CALLM library.
This module contains all prompt templates and prompt-related utilities.
"""

from typing import Dict, Optional


class PromptTemplates:
    """Collection of all prompt templates used in the system."""
    
    # Main generation prompt templates
    GENERATION_TEMPLATES = {
        'alpaca': """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}""",
        
        'chat': """Human: {}
Assistant: {}""",
        
        'simple': "{}",
        
        'instruct': """### Instruction: {}
### Response: {}""",
        
        'llama': """[INST] {} [/INST]
{}""",
        
        'gemma3': """<start_of_turn>user
{}
<end_of_turn>
<start_of_turn>model
{}"""
    }
    
    @classmethod
    def get_generation_template(cls, style: str) -> str:
        """
        Get a generation prompt template by style name.
        
        Args:
            style: The style of prompt template to retrieve
            
        Returns:
            The prompt template string
            
        Raises:
            ValueError: If the style is not found
        """
        if style not in cls.GENERATION_TEMPLATES:
            raise ValueError(f"Unknown prompt style: {style}. Available styles: {list(cls.GENERATION_TEMPLATES.keys())}")
        return cls.GENERATION_TEMPLATES[style]




class PromptBuilder:
    """Utility class for building prompts dynamically."""
    
    @staticmethod
    def build_generation_prompt(instruction: str, style: str = 'alpaca', response: str = "", include_mc_prefix: bool = False) -> str:
        """
        Build a generation prompt with the given instruction and style.
        
        Args:
            instruction: The instruction text
            style: The prompt template style
            response: Optional pre-filled response (for few-shot learning)
            include_mc_prefix: Deprecated (ignored). MC prefix removed entirely.
            
        Returns:
            The formatted prompt string
        """
        template = PromptTemplates.get_generation_template(style)
        
        # Format the template first
        if style in ['simple']:
            formatted_template = template.format(instruction)
        else:
            formatted_template = template.format(instruction, response)
        
        # MC prefix removed completely
        return formatted_template
    
    @staticmethod
    def build_metric_selection_prompt(task_prompt: str) -> str:
        """
        Build a prompt for selecting the appropriate evaluation metric.
        
        Args:
            task_prompt: The task prompt to analyze
            
        Returns:
            The formatted metric selection prompt
        """
        return METRIC_SELECTION_INSTRUCTION.format(prompt=task_prompt)


def list_available_styles() -> list:
    """List all available prompt styles.
    
    Returns:
        List of available prompt style names
    """
    return list(PromptTemplates.GENERATION_TEMPLATES.keys())


METRIC_SELECTION_INSTRUCTION = (
        "Analyze the following task prompt and select the best evaluation metric.\n\n"
        "RULES:\n"
        "1. Code completion tasks (contains 'code', 'python', 'complete the code', etc.) → Use EDIM\n"
        "2. Summarization tasks (contains 'summarize', 'summary', 'write a summary', etc.) → Use ROUGE\n"
        "3. Multiple choice questions (contains 'A)', 'B)', 'C)', 'choose', '判断以下文本', etc.) → Use ACCURACY\n"
        "4. Text simplification tasks (contains 'simplify', 'simplified version', etc.) → Use SARI\n"
        "5. Default for all other tasks → Use ACCURACY\n\n"
        "Task prompt: {prompt}\n\n"
        "Respond with ONLY the metric name. Choose one: ACCURACY, ROUGE, SARI, or EDIM\n\n"
        "### Response: "
)
