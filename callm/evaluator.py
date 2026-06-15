import evaluate
from fuzzywuzzy import fuzz
import re
from typing import Optional


TOKEN_RE = re.compile(
    r"""
    ^\s*
    (?:
        (?P<letter>[A-Za-z])                 # single letter
        (?=[^A-Za-z0-9]|$)                   # must end token immediately (A2 is NOT allowed)
      |
        (?P<number>
            [+-]?\d+(?:\.\d+)?               # integer or decimal
        )
        (?=(?:\.(?!\d))?[^0-9]|$)            # allow trailing '.' only if not a decimal dot
    )
    """,
    re.VERBOSE
)

def extract_single_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = TOKEN_RE.match(text)
    if not m:
        return None
    if m.group('letter'):
        return m.group('letter').upper()
    return m.group('number')

class Evaluator:
    """
        Monitors the performance of each LLM and PEFT based on pre-defined metrics.
        Tracks metrics like accuracy, fluency, coherence, and latency.
        Implements a mechanism to update metrics as new labels become available.
    """
    def __init__(self):
        self.metrics = {}  # Dictionary to store performance metrics
        # Cache expensive metric objects (lazy-loaded on first use)
        self._rouge = None
        self._sari = None

    def _get_rouge(self):
        if self._rouge is None:
            self._rouge = evaluate.load("rouge")
        return self._rouge

    def _get_sari(self):
        if self._sari is None:
            self._sari = evaluate.load("sari")
        return self._sari

    def calculate_sari(self, results):
        inputs = [result["input"] for result in results]
        model_answers = [result["model_answer"] for result in results]
        labels = [[result["ground_truth"]] for result in results]

        sari = self._get_sari()
        sari_score = sari.compute(sources=inputs, predictions=model_answers, references=labels)
        return sari_score['sari']

    def calculate_accuracy(self, results):
        """
        Calculates accuracy for multiple-choice questions by extracting the answer choice 
        (A, B, C, D, etc.) from the model's response or exact-match with single term answers (e.g. 32) 
        and comparing it to the ground truth.
        
        Args:
            results: A list of dictionaries, each containing 'model_answer' and 'ground_truth'.
            
        Returns:
            The accuracy score (float) - percentage of correct answers.
        """
        from callm.utils import Logger
        
        num_correct = 0
        total = 0
        
        Logger.instance().debug(f"\n=== ACCURACY CALCULATION DETAILS ===")
        Logger.instance().debug(f"Evaluating {len(results)} results...")
        # Route verbose per-example logs to details.log with context prefix
        Logger.instance().detail("Showing detailed debug info for first 5 data points only (for sanity check)")
        
        for i, res in enumerate(results):
            model_answer = res["model_answer"]
            ground_truth = res["ground_truth"]
            
            if model_answer is not None:  # Only check if model answer is present
                total += 1
                
                # Extract the multiple-choice answer from the model's response
                extracted_answer = extract_single_answer(model_answer)
                
                # Extract the answer letter from ground truth (handles ScienceQA format)
                extracted_ground_truth = extract_single_answer(ground_truth)
                
                # Log detailed information for first 5 results only (sanity check)
                if i < 5:
                    Logger.instance().detail(f"--- Result {i+1}/{len(results)} (DEBUG SAMPLE) ---")
                    Logger.instance().detail(f"Ground Truth: '{ground_truth}'")
                    Logger.instance().detail(f"Extracted Ground Truth: '{extracted_ground_truth}'")
                    Logger.instance().detail(f"Full Model Output: '{model_answer}'")
                    Logger.instance().detail(f"extract_multiple_choice_answer() output: '{extracted_answer}'")
                
                # Compare extracted answer with extracted ground truth (case-insensitive)
                if extracted_answer and extracted_ground_truth and extracted_answer.upper() == extracted_ground_truth.upper():
                    num_correct += 1
                    if i < 5:  # Only log detailed results for first 5
                        Logger.instance().detail(f"\u2713 CORRECT: Extracted '{extracted_answer}' matches ground truth '{ground_truth}'")
                else:
                    if i < 5:  # Only log detailed results for first 5
                        if extracted_answer:
                            Logger.instance().detail(f"\u2717 INCORRECT: Extracted '{extracted_answer}' != ground truth '{ground_truth}'")
                        else:
                            Logger.instance().detail(f"\u2717 INCORRECT: No answer extracted from model output, ground truth is '{ground_truth}'") 
                    
        if total == 0:
            Logger.instance().debug(f"No valid results found - accuracy = 0.0")
            self.metrics['accuracy'] = 0.0
            return 0.0  # Return 0 if there are no valid results
            
        accuracy = num_correct / total
        Logger.instance().debug(f"\n=== ACCURACY CALCULATION SUMMARY ===")
        Logger.instance().debug(f"Correct: {num_correct}/{total}")
        Logger.instance().debug(f"Final Accuracy Score: {accuracy:.4f}")
        Logger.instance().debug(f"=== END ACCURACY CALCULATION DETAILS ===")
        
        self.metrics['accuracy'] = accuracy
        return accuracy

    def calculate_rouge(self, results):
        model_answers = [result["model_answer"] for result in results]
        ground_truths = [result["ground_truth"] for result in results]

        rouge = self._get_rouge()
        rouge_result = rouge.compute(
            predictions=model_answers,
            references=ground_truths,
            use_aggregator=True,
            use_stemmer=True,
        )
        return rouge_result['rougeL']

    def _postprocess_code_for_edim(self, code):
          code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
          pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
          lits = re.findall(pattern, code)
          for lit in lits:
              code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
          return code


    def calculate_edim(self, results):
        """
        Calculates Edim similarity score for Py150 dataset.
        Args:
             results: A list of dictionaries, each containing 'model_answer' and 'ground_truth'.

        Returns:
              The edim similarity score (float).
        """

        outputs = []
        for output in [result["model_answer"] for result in results]:
            outputs.append(self._postprocess_code_for_edim(output))
        gts = []
        for gt in [result["ground_truth"] for result in results]:
            gts.append(self._postprocess_code_for_edim(gt))
        scores = 0
        valid_count = 0
        for output_id in range(len(outputs)):
            prediction = outputs[output_id]
            target = gts[output_id]
            if prediction == "" or target == "":
                continue
            scores += fuzz.ratio(prediction, target)
            valid_count += 1
        avg_score = scores / valid_count if valid_count > 0 else 0
        return avg_score
