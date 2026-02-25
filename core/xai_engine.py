import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
from models.hf_wrapper import CodeBERTWrapper

class XAIEngine:
    """
    Core engine for zero-shot attribution using Integrated Gradients.
    Maps model decisions back to specific prompt tokens via gradient backpropagation.
    """
    def __init__(self, wrapper: CodeBERTWrapper):
        self.wrapper = wrapper
        target_layer = self.wrapper.model.embeddings.word_embeddings
        self.lig = LayerIntegratedGradients(self._forward_func, target_layer)

    def _forward_func(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, code_cls: torch.Tensor) -> torch.Tensor:
        """Target function y_c: Computes Cosine Similarity between Prompt and Code [CLS]."""
        outputs = self.wrapper.model(input_ids=input_ids, attention_mask=attention_mask)
        prompt_cls = outputs.last_hidden_state[:, 0, :]
        similarity = F.cosine_similarity(prompt_cls, code_cls, dim=1)
        return similarity

    def compute_attribution(self, prompt: str, code: str) -> dict:
        """Computes the attribution matrix for the prompt tokens."""
        code_inputs = self.wrapper.tokenize(code)
        with torch.no_grad():
            code_outputs = self.wrapper.model(**code_inputs)
            code_cls = code_outputs.last_hidden_state[:, 0, :]

        prompt_inputs = self.wrapper.tokenize(prompt)
        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs["attention_mask"]

        baseline_ids = torch.full_like(input_ids, self.wrapper.tokenizer.pad_token_id)

        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, code_cls),
            n_steps=50,
            return_convergence_delta=False
        )
        
        token_attributions = attributions.sum(dim=-1).squeeze(0)
        tokens = self.wrapper.decode_tokens(input_ids)
        
        return {
            "tokens": tokens,
            "attributions": token_attributions.tolist()
        }