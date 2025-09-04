# MTP + Sequence Packing Implementation

## Overview
This implementation enables Multi-Token Prediction (MTP) to work with Sequence Packing, following the approach suggested in the community discussion.

## Changes Made

### 1. Removed Blocking Assertion
**File**: `megatron/core/transformer/multi_token_prediction.py`
- Commented out the assertion that prevented MTP + Sequence Packing combination
- Added explanatory comment about experimental support

### 2. Added Loss Mask Adjustment
**File**: `megatron/core/transformer/multi_token_prediction.py`
- Added `_adjust_loss_mask_for_packed_sequences()` method to `MultiTokenPredictionBlock`
- Automatically masks the last (k+1) tokens of each document in packed sequences
- Prevents cross-document information leakage in MTP predictions

## How It Works

### The Problem
MTP predicts k future tokens at each position. At document boundaries in packed sequences:
- Token i needs to access tokens (i+1) to (i+k) for prediction
- These future tokens might belong to a different document
- This causes incorrect predictions and training signals

### The Solution
For each document in a packed sequence:
1. Identify document boundaries using `PackedSeqParams.cu_seqlens_q`
2. Mask out the last (k+1) tokens of each document in the loss
3. These boundary tokens have no valid future context for MTP

### Example
With MTP depth k=2 and two documents [0,5) and [5,12):
```
Original mask: [1,1,1,1,1,1,1,1,1,1,1,1]
Adjusted mask: [1,1,0,0,0,1,1,1,1,0,0,0]
               ^doc1 last 3^ ^doc2 last 3^
```

## Usage

```python
# Enable MTP with sequence packing
config = TransformerConfig(
    mtp_num_layers=2,  # MTP depth
    mtp_loss_scaling_factor=0.1,
    # ... other config
)

# Create packed sequence parameters
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cumulative_lengths,
    # ... other params
)

# The loss mask adjustment happens automatically in forward pass
# Just pass packed_seq_params to the model as usual
```

## Testing
Run the test script to verify the implementation:
```bash
python test_mtp_loss_mask.py
```

## Notes
- This is experimental support based on community feedback
- The implementation correctly handles variable-length documents
- Loss masking ensures training stability at document boundaries
- Compatible with existing MTP and sequence packing features