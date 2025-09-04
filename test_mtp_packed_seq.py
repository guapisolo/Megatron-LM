#!/usr/bin/env python3
"""Test script to verify MTP + Sequence Packing support"""

import torch
import sys
import os

# Add Megatron-LM to path
sys.path.insert(0, '/root/Megatron-LM')

from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

def test_mtp_with_packed_sequences():
    """Test that MTP works with packed sequences after our modifications"""
    
    print("Testing MTP with Packed Sequences...")
    
    # Create a simple config
    config = TransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        mtp_num_layers=2,  # 2 MTP layers
        mtp_loss_scaling_factor=0.1,
    )
    
    # Create packed sequence parameters
    # Simulate 2 documents: doc1 has 5 tokens, doc2 has 7 tokens
    batch_size = 1
    total_seq_len = 12
    cu_seqlens_q = torch.tensor([0, 5, 12], dtype=torch.int32)  # Cumulative lengths
    
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_q,  # Same for self-attention
        max_seqlen_q=7,  # Maximum document length
        max_seqlen_kv=7,
        qkv_format='thd',  # Format for attention
    )
    
    # Create the MTP block
    try:
        # Get the layer spec
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
        )
        
        # This would normally fail with the assertion, but should work now
        mtp_block = MultiTokenPredictionBlock(
            config=config,
            spec=spec.submodules.mtp[0] if hasattr(spec.submodules, 'mtp') else spec,
        )
        
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (total_seq_len, batch_size))
        position_ids = torch.arange(total_seq_len).unsqueeze(1).expand(-1, batch_size)
        hidden_states = torch.randn(total_seq_len, batch_size, config.hidden_size)
        attention_mask = torch.ones(batch_size, 1, total_seq_len, total_seq_len)
        labels = torch.randint(0, 1000, (total_seq_len, batch_size))
        loss_mask = torch.ones(total_seq_len, batch_size)
        
        # Test the loss mask adjustment
        adjusted_mask = mtp_block._adjust_loss_mask_for_packed_sequences(
            loss_mask, packed_seq_params, config.mtp_num_layers
        )
        
        print(f"Original loss mask shape: {loss_mask.shape}")
        print(f"Adjusted loss mask shape: {adjusted_mask.shape}")
        
        # Check that the last k+1 tokens of each document are masked
        # For doc1 (tokens 0-4): last 3 tokens (2,3,4) should be masked
        # For doc2 (tokens 5-11): last 3 tokens (9,10,11) should be masked
        
        doc1_end_masked = adjusted_mask[2:5, 0]  # Should be zeros
        doc2_end_masked = adjusted_mask[9:12, 0]  # Should be zeros
        
        print(f"\nDoc1 end tokens (should be 0): {doc1_end_masked}")
        print(f"Doc2 end tokens (should be 0): {doc2_end_masked}")
        
        # Verify the masking is correct
        assert torch.all(doc1_end_masked == 0), "Doc1 end tokens not properly masked"
        assert torch.all(doc2_end_masked == 0), "Doc2 end tokens not properly masked"
        
        # Verify non-boundary tokens are not masked
        doc1_start = adjusted_mask[0:2, 0]  # Should be ones
        doc2_start = adjusted_mask[5:9, 0]  # Should be ones
        
        print(f"\nDoc1 start tokens (should be 1): {doc1_start}")
        print(f"Doc2 start tokens (should be 1): {doc2_start}")
        
        assert torch.all(doc1_start == 1), "Doc1 start tokens incorrectly masked"
        assert torch.all(doc2_start == 1), "Doc2 start tokens incorrectly masked"
        
        print("\n✅ All tests passed! MTP now supports sequence packing.")
        
    except AssertionError as e:
        if "multi token prediction + sequence packing" in str(e):
            print(f"❌ Test failed: The assertion is still blocking MTP + packing: {e}")
        else:
            print(f"❌ Test failed with unexpected error: {e}")
            raise
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_mtp_with_packed_sequences()