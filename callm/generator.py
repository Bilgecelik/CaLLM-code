import torch
import gc

from callm.utils import DEVICE, Logger
from callm.memory import MemoryManager
from callm.prompts_repo import PromptTemplates, PromptBuilder


class Generator:
    """
        Executes the selected LLM and PEFT for inference.
        Handles text generation, response formatting, and potential post-processing.
        Can leverage existing libraries like Transformers for efficient inference.
    """

    def __init__(self, config):
        self.config = config
        # Get prompt template from centralized repository
        self.prompt_style = self.config.get('prompt_style', 'alpaca')
        try:
            self.prompt = PromptTemplates.get_generation_template(self.prompt_style)
        except ValueError as e:
            raise NotImplementedError(f"Prompt style '{self.prompt_style}' not implemented. {str(e)}")
        self.model = None
        self.tokenizer = None

    def generate_answers(
        self,
        prompts,
        max_new_tokens: int | None = None,
        include_mc_prefix: bool = False,
        use_cache: bool | None = None,
    ):
        """
        Generates model answers for a batch of prompts.

        Args:
            prompts: A list of prompts (strings).
            max_new_tokens: Optional override for maximum new tokens to generate.
            include_mc_prefix: If True, prepend the MC instruction in the prompt template.
            use_cache: If set, overrides KV-cache usage for generation. If None, uses config default.

        Returns:
            A list of generated answers (strings).
        """

        # Use PromptBuilder to format prompts properly
        formatted_prompts = [
            PromptBuilder.build_generation_prompt(prompt, self.prompt_style, "", include_mc_prefix=include_mc_prefix)
            for prompt in prompts
        ]
        
        # Handle Gemma3 processor: extract inner tokenizer if tokenizer is a processor
        tokenizer_to_use = self.tokenizer
        if hasattr(self.tokenizer, 'tokenizer'):
            # Gemma3Processor has .tokenizer attribute for the actual tokenizer
            tokenizer_to_use = self.tokenizer.tokenizer
        
        inputs = tokenizer_to_use(
            formatted_prompts,
            padding=True,
            max_length=self.config['max_seq_length'],
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        # Default generation length
        if max_new_tokens is None:
            max_new_tokens = int(self.config.get('gen_max_new_tokens', 256))

        # KV-cache: default ON for inference (major speedup)
        if use_cache is None:
            use_cache = bool(self.config.get('gen_use_cache', True))

        # Temperature: if not provided, let the model/generation_config decide
        temperature = self.config.get('temperature', None)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "use_cache": bool(use_cache)}
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            try:
                import torch._dynamo as _dynamo

                @_dynamo.disable
                def _gen_no_compile(model, **kwargs):
                    return model.generate(**kwargs)

                generated_ids = _gen_no_compile(self.model, **inputs, **gen_kwargs)
            except (ImportError, AttributeError) as e:
                Logger.instance().debug(f"Falling back to standard generation due to dynamo issue: {e}")
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                Logger.instance().error(f"Error during model generation: {e}")
                raise
            generated_texts = tokenizer_to_use.batch_decode(generated_ids, skip_special_tokens=True)

        # Clean up memory
        try:
            del inputs
            del generated_ids
        except Exception:
            pass

        return [text.strip() for text in generated_texts]

    def generate_raw(self, prompts, max_new_tokens: int | None = None, use_cache: bool | None = None):
        """
        Generate directly from raw string prompts without applying any prompt template
        or multiple-choice instruction prefix. Useful for meta prompts like metric
        selection.
        """
        
        # Handle Gemma3 processor: extract inner tokenizer if tokenizer is a processor
        tokenizer_to_use = self.tokenizer
        if hasattr(self.tokenizer, 'tokenizer'):
            # Gemma3Processor has .tokenizer attribute for the actual tokenizer
            tokenizer_to_use = self.tokenizer.tokenizer
        
        inputs = tokenizer_to_use(
            prompts,
            padding=True,
            max_length=self.config['max_seq_length'],
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        if max_new_tokens is None:
            max_new_tokens = int(self.config.get('gen_max_new_tokens', 256))

        # KV-cache: default ON for inference
        if use_cache is None:
            use_cache = bool(self.config.get('gen_use_cache', True))

        # Temperature: if not provided, let the model/generation_config decide
        temperature = self.config.get('temperature', None)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "use_cache": bool(use_cache)}
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)

        with torch.no_grad():
            try:
                import torch._dynamo as _dynamo

                @_dynamo.disable
                def _gen_no_compile(model, **kwargs):
                    return model.generate(**kwargs)

                generated_ids = _gen_no_compile(self.model, **inputs, **gen_kwargs)
            except (ImportError, AttributeError) as e:
                Logger.instance().debug(f"Falling back to standard generation in generate_raw: {e}")
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            except Exception as e:
                Logger.instance().error(f"Error during raw model generation: {e}")
                raise

        # Decode and cleanup
        generated_texts = tokenizer_to_use.batch_decode(generated_ids, skip_special_tokens=True)
        try:
            del inputs
            del generated_ids
        except Exception:
            pass

        return [text.strip() for text in generated_texts]

    def extract_answer(self, model_answer: str):
        """
        Extracts the answer from the model's response string.
        Handles multiple occurrences of delimiters by taking the last occurrence.

        Args:
            model_answer: The model's complete response string.

        Returns:
            The extracted answer part (e.g., "A", "B", "C") or None if not found.
        """
        if not model_answer:
            return ""
            
        # Different extraction methods for different prompt styles
        if self.prompt_style == 'alpaca':
            parts = model_answer.split("### Response:")
            if len(parts) > 1:
                # Take everything after the LAST occurrence of "### Response:"
                return parts[-1].strip()
            else:
                # Fallback: return the whole text if no delimiter found
                return model_answer.strip()
        elif self.prompt_style == 'chat':
            parts = model_answer.split("Assistant:")
            if len(parts) > 1:
                # Take everything after the LAST occurrence of "Assistant:"
                return parts[-1].strip()
            else:
                return model_answer.strip()
        elif self.prompt_style == 'llama':
            parts = model_answer.split("[/INST]")
            if len(parts) > 1:
                # Take everything after the LAST occurrence of "[/INST]"
                return parts[-1].strip()
            else:
                return model_answer.strip()
        elif self.prompt_style == 'gemma3':
            # Try multiple possible split patterns for GEMMA3 format
            split_patterns = [
                "<start_of_turn>model\n",  # Most common: with newline
                "<start_of_turn>model",    # Without newline
                "<start_of_turn>model>",   # With closing bracket
                "model\n",                # Just "model" on its own line
                "model"                   # Just "model"
            ]
            
            for pattern in split_patterns:
                parts = model_answer.split(pattern)
                if len(parts) > 1:
                    # Take everything after the LAST occurrence of this pattern
                    extracted = parts[-1].strip()
                    if extracted:  # Only return if we got meaningful content
                        return extracted
            
            # Final fallback: return the whole text
            return model_answer.strip()
        elif self.prompt_style == 'simple':
            # For simple format, the whole response is the answer
            return model_answer.strip()
        else:
            # Default: return the whole response
            return model_answer.strip()

    def generate_evaluation_data(self, batch_data, max_new_tokens: int | None = None):
        """
        Generates the evaluation data with model answers and ground truth.

        Args:
            batch_data: A batch of test data (list of dictionaries).
            max_new_tokens: Optional override for maximum new tokens to generate.

        Returns:
            A list of dictionaries, each containing the model's answer and the ground truth.
        """
        model_answers_list = []
        prompts = [data["prompt"] for data in batch_data]
        generated_texts = self.generate_answers(prompts, max_new_tokens=max_new_tokens)

        for j, data in enumerate(batch_data):
            extracted_answer = self.extract_answer(generated_texts[j])
            ground_truth = data["answer"]
            
            model_answers_list.append(
                {"input": prompts[j], "model_answer": extracted_answer,
                 "ground_truth": ground_truth}
            )

        return model_answers_list
