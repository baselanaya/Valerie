"""
Hybrid CTC/Attention Training Mechanism for Valerie Visual ASR.

This module implements joint CTC and attention-based training with a shared encoder
to improve phoneme prediction accuracy and temporal alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CTCLoss
from typing import Optional, Tuple, List, Dict
import math
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CTCHead(nn.Module):
    """
    CTC head for phoneme prediction with beam search decoding capability.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        vocab_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize CTC head.
        
        Args:
            encoder_dim: Encoder output dimension
            vocab_size: Phoneme vocabulary size (including blank token)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(encoder_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear projection to vocabulary
        self.linear = nn.Linear(encoder_dim, vocab_size)
        
        # CTC loss function
        self.ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        
        logger.debug(f"🔧 CTCHead initialized: encoder_dim={encoder_dim}, vocab_size={vocab_size}")
    
    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CTC head.
        
        Args:
            encoder_outputs: Encoder outputs of shape [B, T, encoder_dim]
            
        Returns:
            CTC logits of shape [B, T, vocab_size]
        """
        # Layer normalization and dropout
        x = self.layer_norm(encoder_outputs)
        x = self.dropout(x)
        
        # Linear projection to vocabulary
        logits = self.linear(x)
        
        return logits
    
    def compute_loss(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss.
        
        Args:
            encoder_outputs: Encoder outputs [B, T, encoder_dim]
            targets: Target phoneme sequences [B, S] (S = max target length)
            input_lengths: Actual input sequence lengths [B]
            target_lengths: Actual target sequence lengths [B]
            
        Returns:
            CTC loss tensor
        """
        # Get CTC logits
        logits = self.forward(encoder_outputs)  # [B, T, V]
        
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # CTC expects [T, B, V] format
        log_probs = log_probs.transpose(0, 1)  # [T, B, V]
        
        # Compute CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss
    
    def decode_greedy(self, encoder_outputs: torch.Tensor) -> List[List[int]]:
        """
        Greedy CTC decoding.
        
        Args:
            encoder_outputs: Encoder outputs [B, T, encoder_dim]
            
        Returns:
            List of decoded phoneme sequences for each batch item
        """
        # Get predictions
        logits = self.forward(encoder_outputs)
        predictions = torch.argmax(logits, dim=-1)  # [B, T]
        
        batch_size, seq_len = predictions.shape
        decoded_sequences = []
        
        for b in range(batch_size):
            sequence = predictions[b].cpu().tolist()
            
            # Remove blanks and consecutive duplicates
            decoded = []
            prev = None
            for token in sequence:
                if token != 0 and token != prev:  # 0 is blank token
                    decoded.append(token)
                prev = token
            
            decoded_sequences.append(decoded)
        
        return decoded_sequences
    
    def decode_beam_search(
        self,
        encoder_outputs: torch.Tensor,
        beam_size: int = 5,
        language_model: Optional[nn.Module] = None
    ) -> List[List[Tuple[List[int], float]]]:
        """
        Beam search CTC decoding.
        
        Args:
            encoder_outputs: Encoder outputs [B, T, encoder_dim]
            beam_size: Beam search width
            language_model: Optional phoneme language model
            
        Returns:
            List of top beam_size hypotheses for each batch item
        """
        # Get log probabilities
        logits = self.forward(encoder_outputs)
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, V]
        
        batch_size, seq_len, vocab_size = log_probs.shape
        batch_results = []
        
        for b in range(batch_size):
            sequence_log_probs = log_probs[b]  # [T, V]
            
            # Initialize beam with empty sequence
            beams = [{'sequence': [], 'score': 0.0, 'last_token': None}]
            
            for t in range(seq_len):
                new_beams = []
                
                for beam in beams:
                    for token in range(vocab_size):
                        token_score = sequence_log_probs[t, token].item()
                        
                        if token == 0:  # Blank token
                            # Continue current sequence without adding token
                            new_beam = {
                                'sequence': beam['sequence'].copy(),
                                'score': beam['score'] + token_score,
                                'last_token': 0
                            }
                            new_beams.append(new_beam)
                        else:
                            # Add token if it's different from the last token
                            if token != beam['last_token']:
                                new_sequence = beam['sequence'] + [token]
                                
                                # Apply language model score if available
                                lm_score = 0.0
                                if language_model is not None:
                                    lm_score = self._get_lm_score(language_model, new_sequence)
                                
                                new_beam = {
                                    'sequence': new_sequence,
                                    'score': beam['score'] + token_score + lm_score,
                                    'last_token': token
                                }
                                new_beams.append(new_beam)
                            else:
                                # Repeat token, keep same sequence
                                new_beam = {
                                    'sequence': beam['sequence'].copy(),
                                    'score': beam['score'] + token_score,
                                    'last_token': token
                                }
                                new_beams.append(new_beam)
                
                # Keep top beam_size beams
                new_beams.sort(key=lambda x: x['score'], reverse=True)
                beams = new_beams[:beam_size]
            
            # Format results
            results = [(beam['sequence'], beam['score']) for beam in beams]
            batch_results.append(results)
        
        return batch_results
    
    def _get_lm_score(self, language_model: nn.Module, sequence: List[int]) -> float:
        """Get language model score for a sequence."""
        # Placeholder for language model scoring
        # In practice, this would use a trained phoneme language model
        return 0.0


class LocationAwareAttention(nn.Module):
    """
    Location-aware attention mechanism for sequence-to-sequence models.
    
    Uses previous attention weights to compute current attention,
    preventing attention from getting stuck on single frames.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int = 256,
        location_conv_filters: int = 32,
        location_conv_kernel: int = 31
    ):
        """
        Initialize location-aware attention.
        
        Args:
            encoder_dim: Encoder hidden dimension
            decoder_dim: Decoder hidden dimension
            attention_dim: Attention mechanism dimension
            location_conv_filters: Number of location convolution filters
            location_conv_kernel: Location convolution kernel size
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        
        # Attention projections
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        
        # Location-based attention
        self.location_conv = nn.Conv1d(
            2, location_conv_filters,  # 2 for cumulative and previous attention
            kernel_size=location_conv_kernel,
            padding=(location_conv_kernel - 1) // 2,
            bias=False
        )
        self.location_proj = nn.Linear(location_conv_filters, attention_dim, bias=False)
        
        # Final attention projection
        self.attention_proj = nn.Linear(attention_dim, 1, bias=False)
        
        # Attention bias
        self.bias = nn.Parameter(torch.randn(attention_dim))
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        attention_weights_cumulative: torch.Tensor,
        attention_weights_previous: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute location-aware attention.
        
        Args:
            encoder_outputs: Encoder outputs [B, T, encoder_dim]
            decoder_hidden: Decoder hidden state [B, decoder_dim]
            attention_weights_cumulative: Cumulative attention weights [B, T]
            attention_weights_previous: Previous attention weights [B, T]
            mask: Optional encoder mask [B, T]
            
        Returns:
            Tuple of (attention_weights, context_vector)
        """
        batch_size, seq_len, encoder_dim = encoder_outputs.shape
        
        # Project encoder outputs
        encoder_proj = self.encoder_proj(encoder_outputs)  # [B, T, attention_dim]
        
        # Project decoder hidden state
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # [B, 1, attention_dim]
        
        # Location-based features
        location_inputs = torch.stack([
            attention_weights_cumulative,
            attention_weights_previous
        ], dim=1)  # [B, 2, T]
        
        location_features = self.location_conv(location_inputs)  # [B, filters, T]
        location_features = location_features.transpose(1, 2)  # [B, T, filters]
        location_proj = self.location_proj(location_features)  # [B, T, attention_dim]
        
        # Compute attention energies
        energies = encoder_proj + decoder_proj + location_proj + self.bias
        energies = torch.tanh(energies)  # [B, T, attention_dim]
        energies = self.attention_proj(energies).squeeze(-1)  # [B, T]
        
        # Apply mask if provided
        if mask is not None:
            energies = energies.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(energies, dim=1)  # [B, T]
        
        # Compute context vector
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),  # [B, 1, T]
            encoder_outputs  # [B, T, encoder_dim]
        ).squeeze(1)  # [B, encoder_dim]
        
        return attention_weights, context_vector


class AttentionDecoder(nn.Module):
    """
    Attention-based decoder for sequence-to-sequence phoneme prediction.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        vocab_size: int,
        attention_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Initialize attention decoder.
        
        Args:
            encoder_dim: Encoder output dimension
            decoder_dim: Decoder hidden dimension
            vocab_size: Phoneme vocabulary size
            attention_dim: Attention mechanism dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer for phonemes
        self.embedding = nn.Embedding(vocab_size, decoder_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=decoder_dim + encoder_dim,  # embedding + context
            hidden_size=decoder_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Location-aware attention
        self.attention = LocationAwareAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim
        )
        
        # Output projection
        self.output_proj = nn.Linear(decoder_dim + encoder_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.debug(f"🔧 AttentionDecoder initialized: decoder_dim={decoder_dim}")
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 200
    ) -> torch.Tensor:
        """
        Forward pass through attention decoder.
        
        Args:
            encoder_outputs: Encoder outputs [B, T, encoder_dim]
            targets: Target sequences for teacher forcing [B, S]
            encoder_mask: Encoder padding mask [B, T]
            max_length: Maximum decoding length for inference
            
        Returns:
            Decoder logits [B, S, vocab_size]
        """
        batch_size, encoder_seq_len, encoder_dim = encoder_outputs.shape
        
        if self.training and targets is not None:
            # Teacher forcing during training
            return self._forward_train(encoder_outputs, targets, encoder_mask)
        else:
            # Autoregressive decoding during inference
            return self._forward_infer(encoder_outputs, encoder_mask, max_length)
    
    def _forward_train(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Training forward pass with teacher forcing."""
        batch_size, target_len = targets.shape
        encoder_seq_len = encoder_outputs.shape[1]
        
        # Initialize decoder state
        hidden = self._init_hidden(batch_size, encoder_outputs.device)
        
        # Initialize attention weights
        attention_weights_cumulative = torch.zeros(
            batch_size, encoder_seq_len, device=encoder_outputs.device
        )
        attention_weights_previous = torch.zeros(
            batch_size, encoder_seq_len, device=encoder_outputs.device
        )
        
        outputs = []
        
        # Add start token (assuming 1 is start token)
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=encoder_outputs.device)
        
        for t in range(target_len):
            # Get current input (either start token or previous target)
            if t > 0:
                decoder_input = targets[:, t-1:t]
            
            # Embed input
            embedded = self.embedding(decoder_input)  # [B, 1, decoder_dim]
            
            # Compute attention
            attention_weights, context_vector = self.attention(
                encoder_outputs,
                hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1],
                attention_weights_cumulative,
                attention_weights_previous,
                encoder_mask
            )
            
            # Update attention weights
            attention_weights_cumulative += attention_weights
            attention_weights_previous = attention_weights
            
            # Concatenate embedding with context
            lstm_input = torch.cat([
                embedded,
                context_vector.unsqueeze(1)
            ], dim=-1)  # [B, 1, decoder_dim + encoder_dim]
            
            # LSTM step
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            
            # Concatenate LSTM output with context for prediction
            output_input = torch.cat([
                lstm_output.squeeze(1),
                context_vector
            ], dim=-1)  # [B, decoder_dim + encoder_dim]
            
            # Project to vocabulary
            output = self.output_proj(self.dropout(output_input))  # [B, vocab_size]
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)  # [B, target_len, vocab_size]
    
    def _forward_infer(
        self,
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 200
    ) -> torch.Tensor:
        """Inference forward pass with autoregressive decoding."""
        batch_size, encoder_seq_len, encoder_dim = encoder_outputs.shape
        
        # Initialize decoder state
        hidden = self._init_hidden(batch_size, encoder_outputs.device)
        
        # Initialize attention weights
        attention_weights_cumulative = torch.zeros(
            batch_size, encoder_seq_len, device=encoder_outputs.device
        )
        attention_weights_previous = torch.zeros(
            batch_size, encoder_seq_len, device=encoder_outputs.device
        )
        
        outputs = []
        
        # Start with start token
        decoder_input = torch.ones(batch_size, 1, dtype=torch.long, device=encoder_outputs.device)
        
        for t in range(max_length):
            # Embed input
            embedded = self.embedding(decoder_input)  # [B, 1, decoder_dim]
            
            # Compute attention
            attention_weights, context_vector = self.attention(
                encoder_outputs,
                hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1],
                attention_weights_cumulative,
                attention_weights_previous,
                encoder_mask
            )
            
            # Update attention weights
            attention_weights_cumulative += attention_weights
            attention_weights_previous = attention_weights
            
            # Concatenate embedding with context
            lstm_input = torch.cat([
                embedded,
                context_vector.unsqueeze(1)
            ], dim=-1)  # [B, 1, decoder_dim + encoder_dim]
            
            # LSTM step
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            
            # Concatenate LSTM output with context for prediction
            output_input = torch.cat([
                lstm_output.squeeze(1),
                context_vector
            ], dim=-1)  # [B, decoder_dim + encoder_dim]
            
            # Project to vocabulary
            output = self.output_proj(self.dropout(output_input))  # [B, vocab_size]
            outputs.append(output)
            
            # Get next input (greedy decoding)
            decoder_input = torch.argmax(output, dim=-1, keepdim=True)  # [B, 1]
            
            # Check for end token (assuming 2 is end token)
            if (decoder_input == 2).all():
                break
        
        return torch.stack(outputs, dim=1)  # [B, decoded_len, vocab_size]
    
    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.decoder_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.decoder_dim, device=device)
        return (h0, c0)


class HybridCTCAttention(nn.Module):
    """
    Hybrid CTC/Attention training mechanism.
    
    Combines CTC and attention losses with configurable weights for
    joint training with a shared encoder.
    """
    
    def __init__(
        self,
        encoder_dim: int,
        vocab_size: int,
        decoder_dim: int = 256,
        attention_dim: int = 256,
        ctc_weight: float = 0.3,
        attention_weight: float = 0.7,
        dropout: float = 0.1
    ):
        """
        Initialize hybrid CTC/Attention module.
        
        Args:
            encoder_dim: Encoder output dimension
            vocab_size: Phoneme vocabulary size
            decoder_dim: Attention decoder dimension
            attention_dim: Attention mechanism dimension
            ctc_weight: Weight for CTC loss
            attention_weight: Weight for attention loss
            dropout: Dropout probability
        """
        super().__init__()
        
        assert abs(ctc_weight + attention_weight - 1.0) < 1e-6, \
            "CTC and attention weights should sum to 1.0"
        
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.attention_weight = attention_weight
        
        # CTC head
        self.ctc_head = CTCHead(encoder_dim, vocab_size, dropout)
        
        # Attention decoder
        self.attention_decoder = AttentionDecoder(
            encoder_dim, decoder_dim, vocab_size, attention_dim, dropout=dropout
        )
        
        # Cross-entropy loss for attention
        self.attention_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        
        logger.info(f"✅ HybridCTCAttention initialized:")
        logger.info(f"   Encoder dimension: {encoder_dim}")
        logger.info(f"   Vocabulary size: {vocab_size}")
        logger.info(f"   CTC weight: {ctc_weight}")
        logger.info(f"   Attention weight: {attention_weight}")
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        ctc_targets: Optional[torch.Tensor] = None,
        attention_targets: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid CTC/Attention module.
        
        Args:
            encoder_outputs: Encoder outputs [B, T, encoder_dim]
            ctc_targets: CTC target sequences [B, S]
            attention_targets: Attention target sequences [B, S]
            input_lengths: Actual input sequence lengths [B]
            target_lengths: Actual target sequence lengths [B]
            encoder_mask: Encoder padding mask [B, T]
            
        Returns:
            Dictionary containing losses and outputs
        """
        results = {}
        
        # CTC branch
        ctc_logits = self.ctc_head(encoder_outputs)
        results['ctc_logits'] = ctc_logits
        
        if self.training and ctc_targets is not None:
            # Compute CTC loss
            ctc_loss = self.ctc_head.compute_loss(
                encoder_outputs, ctc_targets, input_lengths, target_lengths
            )
            results['ctc_loss'] = ctc_loss
        
        # Attention branch
        if self.training and attention_targets is not None:
            attention_logits = self.attention_decoder(
                encoder_outputs, attention_targets, encoder_mask
            )
            results['attention_logits'] = attention_logits
            
            # Compute attention loss
            attention_loss = self.attention_loss(
                attention_logits.reshape(-1, self.vocab_size),
                attention_targets.reshape(-1)
            )
            results['attention_loss'] = attention_loss
            
            # Combined loss
            if 'ctc_loss' in results:
                total_loss = (self.ctc_weight * results['ctc_loss'] + 
                             self.attention_weight * attention_loss)
                results['total_loss'] = total_loss
        else:
            # Inference mode - use attention decoder
            attention_logits = self.attention_decoder(
                encoder_outputs, None, encoder_mask
            )
            results['attention_logits'] = attention_logits
        
        return results
    
    def decode_ctc(self, encoder_outputs: torch.Tensor, beam_size: int = 1) -> List[List[int]]:
        """Decode using CTC."""
        if beam_size == 1:
            return self.ctc_head.decode_greedy(encoder_outputs)
        else:
            return self.ctc_head.decode_beam_search(encoder_outputs, beam_size)
    
    def decode_attention(
        self, 
        encoder_outputs: torch.Tensor, 
        encoder_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode using attention mechanism."""
        return self.attention_decoder(encoder_outputs, None, encoder_mask)


def create_hybrid_ctc_attention(config) -> HybridCTCAttention:
    """
    Factory function to create HybridCTCAttention from configuration.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized HybridCTCAttention instance
    """
    return HybridCTCAttention(
        encoder_dim=config.conformer_dim,
        vocab_size=config.phoneme_vocab_size,
        decoder_dim=config.decoder_dim,
        ctc_weight=getattr(config, 'ctc_weight', 0.3),
        attention_weight=getattr(config, 'attention_weight', 0.7),
        dropout=config.dropout
    )


# Unit tests and examples
if __name__ == "__main__":
    # Test the modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test CTCHead
    print("🧪 Testing CTCHead...")
    ctc_head = CTCHead(encoder_dim=512, vocab_size=40).to(device)
    encoder_outputs = torch.randn(2, 100, 512).to(device)
    ctc_logits = ctc_head(encoder_outputs)
    print(f"✅ CTCHead: {encoder_outputs.shape} -> {ctc_logits.shape}")
    
    # Test greedy decoding
    decoded = ctc_head.decode_greedy(encoder_outputs)
    print(f"✅ CTC greedy decoding: {len(decoded)} sequences")
    
    # Test AttentionDecoder
    print("\n🧪 Testing AttentionDecoder...")
    attention_decoder = AttentionDecoder(
        encoder_dim=512, decoder_dim=256, vocab_size=40
    ).to(device)
    
    # Test with teacher forcing
    targets = torch.randint(1, 40, (2, 50)).to(device)
    attention_logits = attention_decoder(encoder_outputs, targets)
    print(f"✅ AttentionDecoder (training): {encoder_outputs.shape} -> {attention_logits.shape}")
    
    # Test HybridCTCAttention
    print("\n🧪 Testing HybridCTCAttention...")
    hybrid_module = HybridCTCAttention(
        encoder_dim=512, vocab_size=40, decoder_dim=256
    ).to(device)
    
    # Test training mode
    hybrid_module.train()
    ctc_targets = torch.randint(1, 40, (2, 30)).to(device)
    input_lengths = torch.tensor([100, 80]).to(device)
    target_lengths = torch.tensor([30, 25]).to(device)
    
    results = hybrid_module(
        encoder_outputs, ctc_targets, targets, 
        input_lengths, target_lengths
    )
    print(f"✅ HybridCTCAttention (training): {len(results)} outputs")
    
    # Test inference mode
    hybrid_module.eval()
    with torch.no_grad():
        results = hybrid_module(encoder_outputs)
    print(f"✅ HybridCTCAttention (inference): {len(results)} outputs")
    
    print("\n🎉 All tests passed!")
