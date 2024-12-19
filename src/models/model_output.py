
class ModelOutput:
    def __init__(self, final_output, final_output_is_probs, logits_before_activation=None, output_before_ordinal_encoding=None) -> None:
        self.final_output = final_output
        self.final_output_is_probs = final_output_is_probs
        self.logits_before_activation = logits_before_activation
        self.output_before_ordinal_encoding = output_before_ordinal_encoding

    def get_final_output_logits(self):
        assert not self.final_output_is_probs, "Trying to get final model output as logits, but final model output is in probabilities."
        return self.final_output
    
    def get_final_output_probs(self):
        assert self.final_output_is_probs, "Trying to get final model output as probabilities, but final model output is in logits."
        return self.final_output

    def get_probs(self, criterion):
        if self.final_output_is_probs:
            return self.final_output

        assert criterion.convert_to_probs(), "Trying to get output as probabilities but no method to convert."

        return criterion.convert_to_probs(self.final_output)
    
    def get_logits_before_activation(self):
        assert self.logits_before_activation is not None, "Trying to get logits before activation but no activation was used."
        return self.logits_before_activation
    
    def get_output_before_ordinal_encoding(self):
        assert self.output_before_ordinal_encoding is not None, "Trying to get output before ordinal encoding but it doesn't exist."
        return self.output_before_ordinal_encoding
