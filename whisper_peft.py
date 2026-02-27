"""
Custom PEFT wrapper for Whisper models that properly handles input_features
"""
from peft import PeftModelForSeq2SeqLM


class PeftModelForWhisper(PeftModelForSeq2SeqLM):
    """
    Custom PEFT model wrapper for Whisper that uses input_features instead of input_ids
    """
    
    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Forward pass that properly handles Whisper's input_features parameter
        instead of the default input_ids used by PEFT's Seq2SeqLM wrapper.
        """
        peft_config = self.active_peft_config
        
        if not peft_config.is_prompt_learning:
            # Use the PEFT forward hooks but pass through to base_model correctly
            with self._enable_peft_forward_hooks(**kwargs):
                # Filter out special PEFT args
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                
                # Call base_model with Whisper's expected parameters
                return self.base_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )
        else:
            raise ValueError(
                "Prompt learning is not supported in PeftModelForWhisper.forward. "
                "Whisper uses input_features, not input_ids."
            )
    
    def generate(self, input_features=None, attention_mask=None, **kwargs):
        """
        Generate method that properly handles Whisper's input_features parameter
        while maintaining PEFT hooks for proper LoRA functionality.
        
        Args:
            input_features: Input audio features from WhisperFeatureExtractor
            attention_mask: Optional attention mask
            **kwargs: Additional generation parameters (max_length, num_beams, etc.)
        
        Returns:
            Generated token ids
        """
        peft_config = self.active_peft_config
        
        if not peft_config.is_prompt_learning:
            # Use PEFT hooks to ensure LoRA layers are active during generation
            with self._enable_peft_forward_hooks(**kwargs):
                # Filter out special PEFT args
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                
                # Call base_model generate with Whisper's parameters
                outputs = self.base_model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    **kwargs
                )
        else:
            raise ValueError(
                "Prompt learning is not supported in PeftModelForWhisper.generate. "
                "Whisper uses input_features, not input_ids."
            )
        
        return outputs

