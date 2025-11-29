# Generation Parameters Guide: Understanding Model Generation

This guide explains key parameters used in `model.generate()` for controlling text generation behavior, with focus on `do_sample`, `use_cache`, and related parameters.

## Table of Contents

- [Overview](#overview)
- [do_sample: Deterministic vs Stochastic Generation](#do_sample-deterministic-vs-stochastic-generation)
- [use_cache: Key-Value Caching](#use_cache-key-value-caching)
- [Temperature: Controlling Randomness](#temperature-controlling-randomness)
- [Repetition Penalty: Avoiding Repetition](#repetition_penalty-avoiding-repetition)
- [Complete Parameter Reference](#complete-parameter-reference)
- [Common Patterns](#common-patterns)
- [Best Practices](#best-practices)

---

## Overview

When generating text with transformers, you control the generation process through parameters passed to `model.generate()`:

```python
outputs = model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,           # ← Speeds up generation
    max_new_tokens=128,       # ← How many tokens to generate
    do_sample=False,          # ← Deterministic vs random
    temperature=0.0,          # ← Randomness level (if sampling)
    repetition_penalty=1.1    # ← Discourage repetition
)
```

Let's understand each parameter in detail.

---

## do_sample: Deterministic vs Stochastic Generation

### What It Does

`do_sample` controls whether generation is **deterministic** (always the same) or **stochastic** (random/varied).

```python
# Deterministic: Always picks the most likely token
do_sample=False  # Uses greedy decoding

# Stochastic: Samples from probability distribution
do_sample=True   # Uses sampling strategies
```

### How It Works

#### `do_sample=False` (Greedy Decoding)

Always selects the **highest probability** token at each step:

```
Token probabilities at step 1:
- "the" → 0.7  ← ALWAYS CHOSEN
- "a"   → 0.2
- "an"  → 0.1

Token probabilities at step 2:
- "cat" → 0.6  ← ALWAYS CHOSEN
- "dog" → 0.3
- "fox" → 0.1

Result: "the cat" (every time)
```

**Characteristics:**
- ✅ **Deterministic** - Same input = same output
- ✅ **Fast** - No sampling overhead
- ✅ **Coherent** - Follows most likely path
- ❌ **Repetitive** - Can get stuck in loops
- ❌ **Boring** - No creativity or variation

#### `do_sample=True` (Sampling)

Randomly samples tokens based on their probabilities:

```
Token probabilities at step 1:
- "the" → 0.7  ← 70% chance
- "a"   → 0.2  ← 20% chance
- "an"  → 0.1  ← 10% chance

Run 1: Samples "the" (most likely)
Run 2: Samples "a" (less likely but possible)
Run 3: Samples "the" (most likely again)

Result: Different outputs each time!
```

**Characteristics:**
- ✅ **Creative** - More diverse outputs
- ✅ **Varied** - Different results each run
- ✅ **Natural** - More human-like
- ❌ **Non-deterministic** - Can't reproduce exactly
- ❌ **Risky** - May generate nonsense

### Visual Comparison

```
Input: "The weather today is"

╔════════════════════════════════════════════════════════════╗
║ do_sample=False (Greedy)                                   ║
╠════════════════════════════════════════════════════════════╣
║ Run 1: "The weather today is sunny and warm."             ║
║ Run 2: "The weather today is sunny and warm."             ║
║ Run 3: "The weather today is sunny and warm."             ║
║                                                            ║
║ → Always the same (deterministic)                         ║
╚════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════╗
║ do_sample=True (Sampling)                                  ║
╠════════════════════════════════════════════════════════════╣
║ Run 1: "The weather today is sunny and warm."             ║
║ Run 2: "The weather today is cloudy with a chance of rain."║
║ Run 3: "The weather today is perfect for a picnic!"       ║
║                                                            ║
║ → Different each time (stochastic)                        ║
╚════════════════════════════════════════════════════════════╝
```

### When to Use Each

#### Use `do_sample=False` (Greedy) When:

✅ **Factual tasks** - Q&A, information extraction
```python
# Question answering - want consistent, accurate answers
outputs = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=50
)
```

✅ **Code generation** - Want most likely correct code
```python
# Code completion - want reliable, working code
outputs = model.generate(
    **inputs,
    do_sample=False,
    temperature=0.0
)
```

✅ **Reproducibility** - Need same output every time
```python
# Testing, debugging - need consistent behavior
outputs = model.generate(
    **inputs,
    do_sample=False
)
```

✅ **Translation** - Want most accurate translation
```python
# Translation - want best translation, not creative one
outputs = model.generate(
    **inputs,
    do_sample=False
)
```

#### Use `do_sample=True` (Sampling) When:

✅ **Creative writing** - Stories, poems, content
```python
# Story generation - want varied, creative output
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.8,
    top_p=0.9
)
```

✅ **Chatbots** - Want natural, varied responses
```python
# Conversational AI - want human-like variety
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,
    top_k=50
)
```

✅ **Brainstorming** - Want diverse ideas
```python
# Idea generation - want multiple perspectives
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.0,
    num_return_sequences=5  # Generate 5 different ideas
)
```

✅ **Avoiding repetition** - When greedy gets stuck
```python
# When greedy produces repetitive text
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.2
)
```

### Code Examples

#### Example 1: Factual Q&A (Greedy)

```python
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt")

# Use greedy for factual accuracy
outputs = model.generate(
    **inputs,
    do_sample=False,        # Deterministic
    max_new_tokens=20
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Output: "What is the capital of France? The capital of France is Paris."
# (Same answer every time)
```

#### Example 2: Creative Story (Sampling)

```python
prompt = "Once upon a time in a magical forest,"
inputs = tokenizer(prompt, return_tensors="pt")

# Use sampling for creativity
outputs = model.generate(
    **inputs,
    do_sample=True,         # Stochastic
    temperature=0.8,        # Moderate randomness
    top_p=0.9,             # Nucleus sampling
    max_new_tokens=100
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Output: Different creative story each time!
```

---

## use_cache: Key-Value Caching

### What It Does

`use_cache` enables **Key-Value (KV) caching** to speed up autoregressive generation by reusing computed attention values.

```python
use_cache=True   # Enable caching (faster, more memory)
use_cache=False  # Disable caching (slower, less memory)
```

### How It Works

#### Without Caching (`use_cache=False`)

Every token generation recomputes attention for **all previous tokens**:

```
Step 1: Generate "The"
  → Compute attention for: ["The"]

Step 2: Generate "cat"
  → Compute attention for: ["The", "cat"]  ← Recomputes "The"!

Step 3: Generate "sat"
  → Compute attention for: ["The", "cat", "sat"]  ← Recomputes all!

Step 4: Generate "on"
  → Compute attention for: ["The", "cat", "sat", "on"]  ← Recomputes all!

Problem: Redundant computation grows quadratically!
```

#### With Caching (`use_cache=True`)

Stores and reuses attention computations:

```
Step 1: Generate "The"
  → Compute attention for: ["The"]
  → Cache: {key_1, value_1}

Step 2: Generate "cat"
  → Reuse cached: {key_1, value_1}
  → Compute only: {key_2, value_2} for "cat"
  → Cache: {key_1, value_1, key_2, value_2}

Step 3: Generate "sat"
  → Reuse cached: {key_1, value_1, key_2, value_2}
  → Compute only: {key_3, value_3} for "sat"
  → Cache: {key_1, value_1, key_2, value_2, key_3, value_3}

Benefit: Only compute new token's attention!
```

### Performance Impact

```python
# Benchmark: Generate 100 tokens

# Without caching
use_cache=False
# Time: 10.5 seconds
# Memory: 2.1 GB

# With caching
use_cache=True
# Time: 3.2 seconds  ← 3.3x faster!
# Memory: 2.8 GB     ← Uses more memory
```

### Visual Comparison

```
┌─────────────────────────────────────────────────────────┐
│ use_cache=False (No Caching)                            │
├─────────────────────────────────────────────────────────┤
│ Token 1: Compute [1]                                    │
│ Token 2: Compute [1, 2]           ← Recomputes 1       │
│ Token 3: Compute [1, 2, 3]        ← Recomputes 1, 2    │
│ Token 4: Compute [1, 2, 3, 4]     ← Recomputes 1, 2, 3 │
│                                                         │
│ Total computations: 1 + 2 + 3 + 4 = 10                 │
│ Complexity: O(n²)                                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ use_cache=True (With Caching)                           │
├─────────────────────────────────────────────────────────┤
│ Token 1: Compute [1] → Cache                            │
│ Token 2: Reuse [1], Compute [2] → Cache                │
│ Token 3: Reuse [1,2], Compute [3] → Cache              │
│ Token 4: Reuse [1,2,3], Compute [4] → Cache            │
│                                                         │
│ Total computations: 1 + 1 + 1 + 1 = 4                  │
│ Complexity: O(n)                                        │
└─────────────────────────────────────────────────────────┘
```

### When to Use Each

#### Use `use_cache=True` (Default, Recommended)

✅ **Most generation tasks** - Default for good reason
```python
# Standard generation - always use caching
outputs = model.generate(
    **inputs,
    use_cache=True,  # 3-5x faster
    max_new_tokens=100
)
```

✅ **Long sequences** - Speedup increases with length
```python
# Generating long text - caching is essential
outputs = model.generate(
    **inputs,
    use_cache=True,
    max_new_tokens=500  # Much faster with cache
)
```

✅ **Interactive applications** - Need fast response
```python
# Chatbot - users expect quick responses
outputs = model.generate(
    **inputs,
    use_cache=True,
    max_new_tokens=100,
    streamer=streamer
)
```

#### Use `use_cache=False` (Rare)

❌ **Memory constrained** - When you're out of memory
```python
# Very limited memory - sacrifice speed for memory
outputs = model.generate(
    **inputs,
    use_cache=False,  # Saves ~30% memory
    max_new_tokens=50
)
```

❌ **Debugging** - When investigating attention issues
```python
# Debugging attention patterns
outputs = model.generate(
    **inputs,
    use_cache=False,
    output_attentions=True  # Analyze attention
)
```

### Memory vs Speed Trade-off

```
┌──────────────┬──────────┬─────────┬──────────────┐
│ Sequence Len │ use_cache│ Time    │ Memory       │
├──────────────┼──────────┼─────────┼──────────────┤
│ 50 tokens    │ False    │ 2.1s    │ 2.0 GB       │
│ 50 tokens    │ True     │ 0.8s    │ 2.3 GB       │
├──────────────┼──────────┼─────────┼──────────────┤
│ 100 tokens   │ False    │ 8.5s    │ 2.1 GB       │
│ 100 tokens   │ True     │ 1.6s    │ 2.8 GB       │
├──────────────┼──────────┼─────────┼──────────────┤
│ 200 tokens   │ False    │ 34.2s   │ 2.2 GB       │
│ 200 tokens   │ True     │ 3.2s    │ 3.6 GB       │
└──────────────┴──────────┴─────────┴──────────────┘

Conclusion: Cache is almost always worth it!
```

### Code Examples

#### Example 1: Standard Generation (With Cache)

```python
# Recommended: Use caching for normal generation
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    use_cache=True,      # Fast generation
    max_new_tokens=150,
    do_sample=False
)

# Result: Fast, efficient generation
```

#### Example 2: Memory-Constrained (Without Cache)

```python
# Only if you're running out of memory
prompt = "Write a short story:"
inputs = tokenizer(prompt, return_tensors="pt")

try:
    outputs = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=200
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Retrying without cache...")
        outputs = model.generate(
            **inputs,
            use_cache=False,  # Slower but uses less memory
            max_new_tokens=200
        )
```

---

## Temperature: Controlling Randomness

### What It Does

`temperature` controls the **randomness** of sampling when `do_sample=True`. It reshapes the probability distribution.

```python
temperature=0.0   # Deterministic (like greedy)
temperature=0.7   # Moderate randomness (recommended)
temperature=1.0   # Original distribution
temperature=1.5   # High randomness (creative but risky)
```

**Note:** Temperature only matters when `do_sample=True`. When `do_sample=False`, temperature is ignored.

### How It Works

Temperature divides logits (raw model outputs) before applying softmax:

```python
# Original probabilities (temperature=1.0)
"the" → 0.7
"a"   → 0.2
"an"  → 0.1

# Low temperature (0.5) - More confident
"the" → 0.85  ← Even more likely
"a"   → 0.12
"an"  → 0.03

# High temperature (1.5) - More random
"the" → 0.55  ← Less dominant
"a"   → 0.28
"an"  → 0.17
```

### Visual Comparison

```
Temperature = 0.1 (Very Low - Almost Deterministic)
████████████████████ "the" (95%)
█ "a" (4%)
 "an" (1%)

Temperature = 0.7 (Moderate - Recommended)
██████████████ "the" (70%)
████ "a" (20%)
██ "an" (10%)

Temperature = 1.0 (Original Distribution)
████████████ "the" (60%)
██████ "a" (25%)
███ "an" (15%)

Temperature = 2.0 (High - Very Random)
███████ "the" (40%)
██████ "a" (35%)
█████ "an" (25%)
```

### When to Use Different Temperatures

#### Temperature 0.0 - 0.3 (Low - Focused)

✅ **Factual tasks** - Q&A, summaries, translations
```python
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.2,  # Mostly deterministic
    max_new_tokens=100
)
```

#### Temperature 0.5 - 0.8 (Moderate - Balanced)

✅ **Chatbots** - Natural but controlled
```python
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,  # Good balance
    max_new_tokens=100
)
```

#### Temperature 0.9 - 1.2 (High - Creative)

✅ **Creative writing** - Stories, poems
```python
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.0,  # More creative
    max_new_tokens=200
)
```

#### Temperature > 1.5 (Very High - Experimental)

⚠️ **Brainstorming** - Unusual ideas (may be nonsensical)
```python
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=1.8,  # Very random
    max_new_tokens=100
)
```

---

## Repetition Penalty: Avoiding Repetition

### What It Does

`repetition_penalty` discourages the model from repeating tokens it has already generated.

```python
repetition_penalty=1.0   # No penalty (default)
repetition_penalty=1.1   # Slight penalty (recommended)
repetition_penalty=1.5   # Strong penalty
repetition_penalty=2.0   # Very strong penalty (may hurt coherence)
```

### How It Works

Reduces probability of tokens that have already appeared:

```
Without penalty (1.0):
Generated so far: "The cat sat on the"
Next token probabilities:
- "mat" → 0.4
- "the" → 0.3  ← Already used, but no penalty
- "cat" → 0.2  ← Already used, but no penalty
- "floor" → 0.1

With penalty (1.2):
Generated so far: "The cat sat on the"
Next token probabilities:
- "mat" → 0.5    ← Increased (not penalized)
- "the" → 0.2    ← Decreased (penalized)
- "cat" → 0.1    ← Decreased (penalized)
- "floor" → 0.2  ← Increased (not penalized)
```

### When to Use

#### Use `repetition_penalty=1.1-1.3` When:

✅ **Greedy decoding** - Prevent loops
```python
# Greedy often gets stuck in loops
outputs = model.generate(
    **inputs,
    do_sample=False,
    repetition_penalty=1.2,  # Helps avoid "the the the..."
    max_new_tokens=100
)
```

✅ **Long generation** - Maintain variety
```python
# Long text tends to repeat
outputs = model.generate(
    **inputs,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.1,  # Subtle penalty
    max_new_tokens=500
)
```

#### Avoid High Penalties (>1.5)

❌ **Can hurt coherence** - Model avoids necessary repetitions
```python
# Too high penalty can break grammar
outputs = model.generate(
    **inputs,
    repetition_penalty=2.0,  # TOO HIGH
    max_new_tokens=100
)
# May produce: "The cat dog bird fish..." (avoiding "the" hurts grammar)
```

---

## Complete Parameter Reference

### From Lesson 1 Example

```python
outputs = tiny_general_model.generate(
    **inputs,
    streamer=streamer,           # Real-time output
    use_cache=True,              # Speed up generation (3-5x faster)
    max_new_tokens=128,          # Generate up to 128 tokens
    do_sample=False,             # Greedy decoding (deterministic)
    temperature=0.0,             # Ignored when do_sample=False
    repetition_penalty=1.1       # Slight penalty to avoid loops
)
```

### All Common Parameters

```python
outputs = model.generate(
    **inputs,
    
    # Core generation
    max_new_tokens=100,          # How many tokens to generate
    max_length=None,             # Alternative: total length (input + output)
    min_length=0,                # Minimum total length
    
    # Sampling strategy
    do_sample=False,             # True=sampling, False=greedy
    temperature=1.0,             # Randomness (only if do_sample=True)
    top_k=50,                    # Sample from top K tokens
    top_p=1.0,                   # Nucleus sampling threshold
    
    # Repetition control
    repetition_penalty=1.0,      # Penalize repeated tokens
    no_repeat_ngram_size=0,      # Prevent repeating n-grams
    
    # Performance
    use_cache=True,              # Enable KV caching
    
    # Output control
    num_return_sequences=1,      # Generate multiple outputs
    num_beams=1,                 # Beam search (1=no beam search)
    
    # Stopping criteria
    eos_token_id=None,           # Token ID to stop at
    pad_token_id=None,           # Padding token ID
    
    # Streaming
    streamer=None,               # TextStreamer for real-time output
)
```

---

## Common Patterns

### Pattern 1: Factual Q&A (Deterministic)

```python
# Goal: Accurate, consistent answers
outputs = model.generate(
    **inputs,
    do_sample=False,             # Greedy
    use_cache=True,              # Fast
    max_new_tokens=50,
    repetition_penalty=1.1       # Avoid loops
)
```

### Pattern 2: Creative Writing (Sampling)

```python
# Goal: Varied, creative output
outputs = model.generate(
    **inputs,
    do_sample=True,              # Sampling
    temperature=0.8,             # Moderate randomness
    top_p=0.9,                   # Nucleus sampling
    use_cache=True,              # Fast
    max_new_tokens=200,
    repetition_penalty=1.1       # Avoid repetition
)
```

### Pattern 3: Chatbot (Balanced)

```python
# Goal: Natural, varied, but coherent
outputs = model.generate(
    **inputs,
    do_sample=True,              # Sampling for variety
    temperature=0.7,             # Moderate randomness
    top_p=0.9,                   # Nucleus sampling
    use_cache=True,              # Fast response
    max_new_tokens=100,
    repetition_penalty=1.1,      # Avoid repetition
    streamer=streamer            # Real-time output
)
```

### Pattern 4: Code Generation (Conservative)

```python
# Goal: Correct, working code
outputs = model.generate(
    **inputs,
    do_sample=True,              # Slight sampling
    temperature=0.2,             # Very low randomness
    use_cache=True,              # Fast
    max_new_tokens=200,
    repetition_penalty=1.0       # No penalty (code may repeat)
)
```

---

## Best Practices

### 1. **Always Use Caching (Unless Memory-Constrained)**

```python
# ✅ Good: Fast generation
use_cache=True

# ❌ Bad: 3-5x slower for no benefit
use_cache=False  # Only if out of memory
```

### 2. **Match Temperature to do_sample**

```python
# ✅ Good: Consistent settings
do_sample=False
temperature=0.0  # Ignored anyway

# ✅ Good: Sampling with temperature
do_sample=True
temperature=0.7

# ❌ Confusing: Temperature ignored
do_sample=False
temperature=0.9  # Has no effect!
```

### 3. **Use Moderate Repetition Penalties**

```python
# ✅ Good: Subtle penalty
repetition_penalty=1.1

# ✅ Good: Moderate penalty
repetition_penalty=1.2

# ❌ Bad: Too high, hurts coherence
repetition_penalty=2.0
```

### 4. **Choose do_sample Based on Task**

```python
# ✅ Factual tasks: Greedy
do_sample=False

# ✅ Creative tasks: Sampling
do_sample=True
temperature=0.7-0.9
```

### 5. **Set Reasonable max_new_tokens**

```python
# ✅ Good: Reasonable limits
max_new_tokens=100  # Short response
max_new_tokens=500  # Long response

# ❌ Bad: Unnecessarily large
max_new_tokens=10000  # Wastes time/memory
```

---

## Summary

### Quick Reference Table

| Parameter | Purpose | Recommended Value | When to Change |
|-----------|---------|-------------------|----------------|
| `do_sample` | Deterministic vs random | `False` for facts, `True` for creativity | Based on task type |
| `use_cache` | Speed up generation | `True` (always) | Only if out of memory |
| `temperature` | Control randomness | `0.7` (if sampling) | Higher for creativity, lower for accuracy |
| `repetition_penalty` | Avoid repetition | `1.1` | Higher if seeing loops |
| `max_new_tokens` | Length limit | Task-dependent | Based on desired output length |

### Key Takeaways

1. **`do_sample=False`** → Deterministic, consistent, good for facts
2. **`do_sample=True`** → Random, creative, good for variety
3. **`use_cache=True`** → Always use (3-5x faster)
4. **`temperature`** → Only matters when `do_sample=True`
5. **`repetition_penalty=1.1`** → Subtle penalty prevents loops

### Lesson 1 Settings Explained

```python
outputs = tiny_general_model.generate(
    **inputs,
    streamer=streamer,           # Show output in real-time
    use_cache=True,              # 3-5x faster (always use)
    max_new_tokens=128,          # Generate up to 128 tokens
    do_sample=False,             # Greedy = deterministic output
    temperature=0.0,             # Ignored (do_sample=False)
    repetition_penalty=1.1       # Slight penalty to avoid loops
)
```

**Why these settings?**
- **Greedy decoding** (`do_sample=False`) for consistent, predictable output
- **Caching enabled** for fast generation
- **Small repetition penalty** to prevent getting stuck in loops
- **Temperature=0.0** has no effect (greedy ignores it)

---

## Further Reading

- [HuggingFace Generation Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [Generation Strategies Guide](https://huggingface.co/docs/transformers/generation_strategies)
- [Text Streaming Guide](text-streaming-guide.md)
- [Model Downloading Guide](../model-downloading-guide.md)

## Related Topics

- **Beam Search** - Alternative to greedy/sampling
- **Top-k and Top-p Sampling** - Advanced sampling strategies
- **Length Penalties** - Control output length
- **Constrained Generation** - Force specific patterns
