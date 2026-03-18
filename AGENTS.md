# NVFP4 MMA Emulation Framework

## Project Overview

This project provides bit-accurate emulation of NVIDIA Blackwell NVFP4 MMA instructions, 
with a focus on modeling precision loss during hardware reduction operations.

## Architecture

```
emulation/
├── __init__.py          # Package exports
├── config.py            # HardwareConfig, RoundStrategy
├── core.py              # HardwareCore, MMAEngine (5-stage pipeline)
├── strategies.py        # Rounding implementations
├── emulator.py          # High-level MMAEmulator API
├── utils.py             # NVFP4Utils, DataGenerator
├── search.py            # Parameter search functionality
└── configs/             # Hardware-specific configurations
    └── NVFP4_OFP16_5090.json
```

## Key Findings

### Successful Configuration for RTX 5090

```json
{
  "w_stage3": 34,        // Intra-block (4-to-1) reduction bit width
  "w_stage4": 28,        // Inter-block accumulation bit width
  "stage3_rounding": "RZ",
  "stage4_rounding": "RZ",
  "block0_cast_rounding": "RZ",
  "output_rounding": "RNE"
}
```

### Critical Insights

1. **Stage 3 needs HIGHER precision (W=28)** than Stage 4 (W=24)
   - Stage 3: 4-to-1 reduction within K=64 block
   - Stage 4: Inter-block accumulation with FP32 accumulator

2. **Asymmetric truncation strategy**:
   - Block 0: Cast to FP32 (full precision accumulator)
   - Block 1, 2, ...: Keep W-bit precision
   - Accumulation: `hardware_add_inter_block(FP32_acc, W_bits_new_val)`

3. **Negative numbers require different parameters**:
   - Positive-only: W3=28, W4=24 works
   - With negatives: May need W3=28, W4=28 or higher

### Why This Works

The key insight is that hardware uses:
1. **High precision for intra-block reduction** (Stage 3) because all 4 groups are reduced together
2. **Lower precision for inter-block** (Stage 4) because the FP32 accumulator provides sufficient headroom

## 5-Stage Pipeline

```
Stage 1: Intra-Group Partial Sum (Lossless FP16)
  Input: FP4 unpacked to FP16
  Output: [M, N, G] partial sums

Stage 2: Apply Shared Scales (Lossless F32)
  Input: FP16 partial sums
  Output: [M, N, G] scaled partials (FP32)

Stage 3: Group -> MMA-K-Block Reduction (W3-bit)
  Input: [M, N, G] scaled partials
  Output: [M, N, num_blocks] block sums (W-bit or FP32)
  
Stage 4: Inter-Block Accumulation (W4-bit)
  Input: Block sums (Block 0=FP32, others=W-bit)
  Output: [M, N] accumulated result (FP32)

Stage 5: Final Cast (FP16)
  Input: FP32 accumulated result
  Output: FP16 final output
```

## Usage Examples

### Basic Emulation

```python
from emulation import MMAEmulator, HardwareConfig

config = HardwareConfig.from_json("emulation/configs/NVFP4_OFP16_5090.json")
emulator = MMAEmulator(config)

result = emulator.run(a_fp4, b_fp4, scale_a, scale_b, alpha)
```

### Parameter Search

```python
from emulation.search import ParameterSearch

searcher = ParameterSearch(num_iterations=1000)
results = searcher.grid_search(
    w_stage3_range=range(25, 35),
    w_stage4_range=range(20, 30)
)

# Get best config
best_config = results[0].config
```

### Quick Search

```python
from emulation.search import quick_search

config = quick_search()
emulator = MMAEmulator(config)
```

## Development Notes

### Context Window Management

- Key parameters are documented in configs/ directory
- This AGENTS.md file preserves critical findings
- Code is modular to allow independent discussion of components

### Testing Strategy

1. **Unit tests**: Each stage independently
2. **Integration tests**: Full pipeline with known inputs
3. **Regression tests**: Ensure found configs remain valid
4. **Corner cases**: Negative numbers, edge values, different K values

### Future Work

- [ ] Optimize chunking for very large matrices
- [ ] Support BF16 output
- [ ] Add support for sparse MMA
- [ ] Implement gradient computation for training
- [ ] Add visualization tools for error analysis

## References

- CUTLASS 3.8 documentation
- NVIDIA Blackwell architecture whitepaper
- NVFP4 format specification (E2M1)
