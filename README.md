# wave32-cycles-disp09 (DISPATCH-09 REV=2)

Single-file, zero-dependency Python 3.12+ GFX11 (RDNA3 Wave32) raw binary ISA decoder, 32-bank LDS & multi-wave CU cycle-exact emulator, compact binary/hex SQTT nibble-packet codec, and least-squares `fit_profile` latency calibrator.

## Verification
```bash
python3 -W error wave32_cycles.py --self-test
```
All 30/30 deterministic unit tests pass with 0 warnings (`fit_profile` recovers hidden non-default `LatencyProfile` with `0.000000` MAE).
