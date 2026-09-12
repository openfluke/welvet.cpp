# Native dtype training and learning-rate examples

[Open the results and learning curves](comparison.md).

This folder is separate from the dense timing benchmarks. It measures actual
learning: predictions determine errors, errors determine fresh native gradients,
and weights change before the next example. It does not replay fixed gradients.

## The shared task

One dense layer has 16 bipolar inputs and 16 binary output decisions, with no
bias. A fixed random ±1 teacher generates labels by thresholding its dense sums
at zero (ties are +1). This is a synthetic multi-output classification task:
the teacher includes exact zero-score cases labeled +1. This favors recovery of
the discrete teacher and does not establish that binary is generally more accurate
than floating point. Here, accuracy is the fraction of output bits predicted correctly, not the fraction of
examples with all 16 labels correct.

Every dtype and language receives exactly the same logical examples, teacher
labels and initial weights. There are 256 training, 128 validation and 256 test
examples; the input sets do not overlap. Initialization seeds 11, 22 and 33 are
shared across dtypes. The dataset is fixed; these seeds vary initialization, not
the train/test split. A separate seed 99 is used only for speed calibration.
Training visits the same deterministic example order each epoch.

The sweep tests LR 1/1024, 1/256, 1/64, 1/32, 1/8, 1/2, 1 and 2 for 32 epochs
(8,192 online steps). The report selects an LR by mean final validation error,
then earlier 90% validation attainment, then smaller LR. Test accuracy and timing
are never used to select the learning rate. Three initializations on one synthetic
task cannot establish a universal “right learning rate” for a dtype.

## What the learner does

For each training example:

1. Native forward computes scores or packed binary/ternary outputs.
2. Compare output signs with the target labels. Zero scores predict +1.
3. For a mistaken output, set upstream direction to `-target`. Correct outputs
   receive a representable dummy direction, and their optimizer rows are skipped.
4. Compute both dX and dW using the real native-width/packed outer-product kernels.
   dX is retained as observable work, though this single-layer model does not need
   it for another layer.
5. Update only mistaken output rows, using their freshly computed dW.

This is an online mistake-driven perceptron rule, with quantized updates for
integer/binary/ternary weights. It is not a demonstration of multi-layer binary
backpropagation or the exact derivative of a sign threshold.

The output-level no-update mask is explicit and is used for **all** dtypes. It
solves binary's inability to encode a zero gradient without widening the binary
gradient tensor. It is separate control state, not a hidden floating-point
activation or per-weight shadow model. There is one boolean flag per output.

## Native arithmetic and unsigned encoding

Weights, inputs, outputs, upstream gradients, dX and dW use the named dtype.
Binary uses one packed bit per value; ternary uses packed nonzero/sign planes.
Numeric integer arithmetic wraps at its own width. There are no floating-point
gradient tensors for discrete dtypes.

The common task uses logical ±1. For unsigned types, logical −1 is represented by
the all-ones bit pattern, and the score's bit pattern is interpreted as a
same-width signed value for classification. All matrix arithmetic remains
unsigned/modular. Consequently, signed/unsigned counterparts with the same width
can have identical learning behavior here. This is an explicit two's-complement
encoding of the common task; it is not ordinary nonnegative-real arithmetic.

## Learning rate and buffer

Floating-point weights apply the LR directly in FP32/FP64. Inputs and initial
weights are integers and tested LRs are dyadic fractions, so these small runs also
permit exact cross-language checks.

Discrete residuals are signed int64 Q16; every tested LR is exact. For gradient
direction `g ∈ {-1,+1}` on a mistaken row:

```
u = residual - lr_q16 * g
quantum = 131072 for binary, 65536 otherwise
movement = sign(u) * floor((abs(u) + quantum/2 - 1) / quantum)
```

Integer weights apply `movement` with same-width wrap. Binary flips have size 2;
ternary steps have size 1, both saturating to their legal values. Subtract the
actual applied movement from the residual; retain overshoot, but discard outward
residual at binary/ternary endpoints. Exact midpoint ties hold. This differs from
the earlier microbenchmark optimizer's saturating integer updates and fixed LR.
The new rule is stated explicitly so the training results are reproducible.

The unbuffered control discards the previous discrete residual immediately before
each active update. It retains the same LR, inputs, initialization, backward
arithmetic and stopping rule. At LR 1/32, an individual request is too small to
change an integer/binary/ternary weight without accumulation. Floating-point
updates are identical in the two controls because they already accept fractional
steps. An unused int64 residual array remains allocated in floating-point runs for
a uniform driver layout; it is not used to update those weights.

## Run the examples

```bash
# Generate/build, independently verify, sweep and render all results:
python3 training_lr/run.py

# Use existing binaries but still run correctness checks:
python3 training_lr/run.py --skip-build

# A single human-readable example:
python3 training_lr/example.py --dtype binary --language Go --lr 1/32
python3 training_lr/example.py --dtype int8 --language Rust --lr 1/64
python3 training_lr/example.py --dtype float32 --language C++ --lr 1/8

# Matched binary control:
python3 training_lr/example.py --dtype binary --language Go --lr 1/32 --no-buffer
```

The example wrapper accepts `--epochs` and `--seed`. It reports the effective Q16
learning rate, progress, test accuracy and training time. The full sweep publishes
all tested rates, not just the winners. Generated fixtures/binaries and example
state files live under the ignored `native_arithmetic/build/training-lr` folder.

## Timing and checks

Each language/dtype chooses dot or tile forward kernels using a separate
initialization; C++ also compares GCC and Clang. Backward uses the native outer
kernels. Selection uses three timing samples of eight training epochs at LR 1/32,
without looking at accuracy. The full LR sweep is separate from that calibration.
All timed processes use one thread pinned to core 2 by default.

Training time includes forward, error detection, both backward products and
updates. It excludes setup, evaluation, process startup and file I/O. An
additional wall-time field includes evaluation. Validation is checked every four
epochs, so reported times to 90% are checkpoint bounds rather than exact crossing
times. Median times over three initializations reflect this small full trainer,
not peak large-matrix throughput.

`check.py` compares an independent scalar training reference against both forward
modes in all compiler builds. It checks 192 dtype/LR/buffer combinations, learning
curves, held-out errors, weights and residuals, plus a binary flip-and-stop case.
The existing kernels have their own larger arithmetic tests. Every full sweep
case must also match all three languages in its learning curve and final state.

`results.json` contains all 1,728 sweep runs, calibration timings, the logical
dataset, selections and source/fixture/binary hashes. `comparison.md` presents the
results; PNG plots show trajectories and the matched buffer control.

Driver templates are in `templates/`. `build.py` generates native Go/Rust/C++
training drivers using the tuned kernel sources. This experiment does not edit
Welvet or change the earlier benchmark results.
