import torch
from transformers import AutoTokenizer, AutoModel

class CodeBERTWrapper:
    """
    Wrapper for microsoft/codebert-base to process text and code tokens.
    Handles device mapping and tensor tokenization.
    """
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Freeze layers, critical for stable gradient computation

    def tokenize(self, text: str) -> dict:
        """Tokenizes input string into PyTorch tensors."""
        return self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)

    def decode_tokens(self, input_ids: torch.Tensor) -> list:
        """Converts token IDs back to string representations for mapping."""
        return self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())