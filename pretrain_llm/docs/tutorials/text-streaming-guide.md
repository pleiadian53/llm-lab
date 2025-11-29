# Text Streaming Guide: Understanding TextStreamer

This guide explains the `TextStreamer` class from HuggingFace Transformers and its use cases for real-time text generation.

## Table of Contents

- [What is TextStreamer?](#what-is-textstreamer)
- [Why Use Streaming?](#why-use-streaming)
- [Basic Usage](#basic-usage)
- [Configuration Options](#configuration-options)
- [Use Cases](#use-cases)
- [Advanced Patterns](#advanced-patterns)
- [Performance Considerations](#performance-considerations)

---

## What is TextStreamer?

`TextStreamer` is a callback class in the HuggingFace Transformers library that enables **real-time token-by-token output** during text generation. Instead of waiting for the entire generation to complete, it prints tokens as they're generated.

### The Problem It Solves

**Without streaming:**
```python
# User waits... (5-10 seconds for long text)
# Then suddenly sees: "The quick brown fox jumps over the lazy dog..."
```

**With streaming:**
```python
# User sees tokens appear in real-time:
# "The" → "quick" → "brown" → "fox" → "jumps" → ...
```

### How It Works

```
┌─────────────┐
│   Model     │
│ Generation  │
└──────┬──────┘
       │ Generates tokens one by one
       ▼
┌─────────────┐
│TextStreamer │ ◄── Receives each token
│             │     Decodes it
│             │     Prints immediately
└─────────────┘
```

---

## Why Use Streaming?

### 1. **Better User Experience**

Users see progress immediately instead of waiting for completion:

```python
# Without streaming - user sees nothing for 10 seconds
output = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))  # All at once after 10s

# With streaming - user sees tokens appear progressively
streamer = TextStreamer(tokenizer)
model.generate(inputs, max_new_tokens=100, streamer=streamer)
# "The" (0.1s) → "quick" (0.2s) → "brown" (0.3s) → ...
```

### 2. **Interactive Applications**

Essential for chatbots, assistants, and interactive demos:

```python
# Chatbot feels responsive
User: "Tell me about machine learning"
Bot: "Machine" → "learning" → "is" → "a" → "subset" → ...
     (User can start reading immediately)
```

### 3. **Long-Form Content**

For generating long documents, users can start reading while generation continues:

```python
# Generate a 500-word essay
# User can read paragraph 1 while paragraphs 2-5 are still generating
```

### 4. **Debugging and Monitoring**

See what the model is generating in real-time to catch issues early:

```python
# If model starts generating nonsense, you can stop it early
# Instead of waiting for 1000 tokens of garbage
```

---

## Basic Usage

### Minimal Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create streamer
streamer = TextStreamer(tokenizer)

# Generate with streaming
inputs = tokenizer("Once upon a time", return_tensors="pt")
model.generate(**inputs, max_new_tokens=50, streamer=streamer)
```

**Output (appears progressively):**
```
Once upon a time, there was a young girl named Alice who lived in a small village...
```

### From Lesson 1 Example

```python
from transformers import TextStreamer

streamer = TextStreamer(
    tiny_general_tokenizer,
    skip_prompt=True,              # Don't repeat the input prompt
    skip_special_tokens=True       # Don't show <BOS>, <EOS>, etc.
)

# Use in generation
outputs = tiny_general_model.generate(
    **inputs,
    streamer=streamer,             # Enable streaming
    max_new_tokens=50
)
```

---

## Configuration Options

### 1. `skip_prompt` (bool, default=False)

Controls whether to print the input prompt.

**`skip_prompt=False`** (default):
```python
streamer = TextStreamer(tokenizer, skip_prompt=False)
# Input: "Hello, my name is"
# Output: "Hello, my name is Alice and I live in..."
#         ^^^^^^^^^^^^^^^^^^^^ (prompt repeated)
```

**`skip_prompt=True`** (recommended):
```python
streamer = TextStreamer(tokenizer, skip_prompt=True)
# Input: "Hello, my name is"
# Output: " Alice and I live in..."
#         (only new tokens)
```

**Use case:** Set to `True` when you've already shown the prompt to the user.

### 2. `skip_special_tokens` (bool, default=False)

Controls whether to show special tokens like `<BOS>`, `<EOS>`, `<PAD>`.

**`skip_special_tokens=False`** (default):
```python
streamer = TextStreamer(tokenizer, skip_special_tokens=False)
# Output: "<BOS>Hello world<EOS>"
```

**`skip_special_tokens=True`** (recommended):
```python
streamer = TextStreamer(tokenizer, skip_special_tokens=True)
# Output: "Hello world"
```

**Use case:** Set to `True` for cleaner output in user-facing applications.

### 3. `decode_kwargs` (dict, optional)

Additional arguments passed to `tokenizer.decode()`.

```python
streamer = TextStreamer(
    tokenizer,
    skip_special_tokens=True,
    decode_kwargs={
        "clean_up_tokenization_spaces": True,  # Remove extra spaces
        "skip_special_tokens": True,           # Can also set here
    }
)
```

---

## Use Cases

### Use Case 1: Interactive Chat

```python
def chat_with_streaming(user_message):
    """Interactive chatbot with streaming responses."""
    prompt = f"User: {user_message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    print("Assistant: ", end="", flush=True)
    model.generate(
        **inputs,
        max_new_tokens=100,
        streamer=streamer,
        do_sample=True,
        temperature=0.7
    )
    print()  # New line after generation

# Usage
chat_with_streaming("What is machine learning?")
# Output appears progressively:
# Assistant: Machine learning is a subset of artificial intelligence...
```

### Use Case 2: Story Generation

```python
def generate_story_with_streaming(prompt):
    """Generate a story with real-time output."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("Story:\n")
    model.generate(
        **inputs,
        max_new_tokens=500,  # Long story
        streamer=streamer,
        do_sample=True,
        top_p=0.9,
        temperature=0.8
    )

# Usage
generate_story_with_streaming("Once upon a time in a magical forest,")
# User can start reading while generation continues
```

### Use Case 3: Code Generation with Monitoring

```python
def generate_code_with_streaming(description):
    """Generate code and monitor for errors."""
    prompt = f"# Task: {description}\n# Python code:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("Generated Code:")
    model.generate(
        **inputs,
        max_new_tokens=200,
        streamer=streamer,
        temperature=0.2  # Lower temperature for code
    )

# Usage
generate_code_with_streaming("Create a function to calculate fibonacci numbers")
# Watch code appear line by line
```

### Use Case 4: Translation with Progress

```python
def translate_with_streaming(text, target_lang="French"):
    """Translate text with streaming output."""
    prompt = f"Translate to {target_lang}: {text}\nTranslation:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    print(f"Translation ({target_lang}): ", end="", flush=True)
    model.generate(**inputs, max_new_tokens=100, streamer=streamer)
    print()

# Usage
translate_with_streaming("Hello, how are you today?")
# Translation appears word by word
```

---

## Advanced Patterns

### Custom Streamer

You can create custom streamers for specific behaviors:

```python
from transformers import TextStreamer

class CustomStreamer(TextStreamer):
    """Custom streamer with additional features."""
    
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_count = 0
        self.generated_text = []
    
    def put(self, value):
        """Called for each generated token."""
        self.token_count += 1
        super().put(value)  # Call parent to print
    
    def end(self):
        """Called when generation is complete."""
        super().end()
        print(f"\n[Generated {self.token_count} tokens]")

# Usage
streamer = CustomStreamer(tokenizer, skip_prompt=True)
model.generate(**inputs, max_new_tokens=50, streamer=streamer)
# Output: ...generated text...
#         [Generated 50 tokens]
```

### Streaming to File

```python
class FileStreamer(TextStreamer):
    """Stream output to a file."""
    
    def __init__(self, tokenizer, filepath, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.file = open(filepath, 'w')
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Override to write to file instead of stdout."""
        self.file.write(text)
        self.file.flush()
        if stream_end:
            self.file.close()

# Usage
streamer = FileStreamer(tokenizer, "output.txt", skip_prompt=True)
model.generate(**inputs, max_new_tokens=100, streamer=streamer)
# Output written to output.txt in real-time
```

### Streaming with Callbacks

```python
class CallbackStreamer(TextStreamer):
    """Stream with custom callback function."""
    
    def __init__(self, tokenizer, callback, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.callback = callback
    
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Call custom callback with each text chunk."""
        super().on_finalized_text(text, stream_end)
        self.callback(text, stream_end)

# Usage
def my_callback(text, is_end):
    # Send to websocket, update UI, etc.
    print(f"[Callback received: {len(text)} chars]")

streamer = CallbackStreamer(tokenizer, my_callback, skip_prompt=True)
model.generate(**inputs, max_new_tokens=50, streamer=streamer)
```

---

## Performance Considerations

### 1. **Overhead**

Streaming adds minimal overhead (~1-5% slower) due to decoding each token:

```python
# Without streaming: Decode once at the end
# Time: 10.0 seconds

# With streaming: Decode each token
# Time: 10.2 seconds (2% overhead)
```

**Verdict:** The UX improvement far outweighs the small performance cost.

### 2. **Batch Generation**

`TextStreamer` works with batch size 1. For batch generation:

```python
# Streaming works
model.generate(**inputs, max_new_tokens=50, streamer=streamer)  # batch_size=1

# Streaming doesn't make sense for batches
inputs_batch = tokenizer(["prompt1", "prompt2"], return_tensors="pt", padding=True)
model.generate(**inputs_batch, max_new_tokens=50)  # No streamer
```

### 3. **Token Buffering**

Some tokens are buffered before printing to avoid partial words:

```python
# Model generates: ["The", "##quick", "##brown"]
# Streamer outputs: "The" → "quick" → "brown"
#                   (waits for complete words)
```

### 4. **Memory Usage**

Streaming doesn't reduce memory usage - the full generation is still stored:

```python
# Both use the same memory
outputs = model.generate(**inputs, max_new_tokens=100)  # No streaming
outputs = model.generate(**inputs, max_new_tokens=100, streamer=streamer)  # Streaming

# Memory is determined by model size and generation length, not streaming
```

---

## Comparison: Streaming vs Non-Streaming

| Aspect | Non-Streaming | Streaming |
|--------|---------------|-----------|
| **User Experience** | Wait for completion | See progress immediately |
| **Latency to First Token** | High (wait for all) | Low (see first token quickly) |
| **Total Generation Time** | Baseline | +1-5% overhead |
| **Memory Usage** | Same | Same |
| **Debugging** | See output at end | Monitor in real-time |
| **Use Case** | Batch processing | Interactive applications |
| **Code Complexity** | Simpler | Slightly more complex |

---

## Best Practices

### 1. **Always Skip Prompt in Interactive Apps**

```python
# ✅ Good: User doesn't see their input repeated
streamer = TextStreamer(tokenizer, skip_prompt=True)

# ❌ Bad: User sees their input echoed back
streamer = TextStreamer(tokenizer, skip_prompt=False)
```

### 2. **Skip Special Tokens for Clean Output**

```python
# ✅ Good: Clean, readable output
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# ❌ Bad: User sees <BOS>, <EOS>, etc.
streamer = TextStreamer(tokenizer, skip_special_tokens=False)
```

### 3. **Use Lower Temperature for Streaming**

Streaming makes errors more visible, so use conservative generation:

```python
# ✅ Good: More coherent, fewer errors
model.generate(
    **inputs,
    streamer=streamer,
    temperature=0.7,      # Moderate randomness
    top_p=0.9,
    repetition_penalty=1.1
)

# ❌ Risky: High randomness can produce visible nonsense
model.generate(
    **inputs,
    streamer=streamer,
    temperature=1.5,      # Too random
    do_sample=True
)
```

### 4. **Add Visual Indicators**

```python
# ✅ Good: User knows generation is happening
print("Assistant: ", end="", flush=True)
model.generate(**inputs, streamer=streamer)
print("\n[Generation complete]")

# ❌ Bad: No context for the streaming text
model.generate(**inputs, streamer=streamer)
```

---

## Common Pitfalls

### Pitfall 1: Forgetting `flush=True`

```python
# ❌ Bad: Output may be buffered
print("Assistant: ", end="")
model.generate(**inputs, streamer=streamer)

# ✅ Good: Force immediate output
print("Assistant: ", end="", flush=True)
model.generate(**inputs, streamer=streamer)
```

### Pitfall 2: Using Streaming with Batches

```python
# ❌ Bad: Streaming doesn't work well with batches
inputs = tokenizer(["prompt1", "prompt2"], return_tensors="pt", padding=True)
model.generate(**inputs, streamer=streamer)  # Confusing output

# ✅ Good: Use streaming for single prompts
inputs = tokenizer("prompt1", return_tensors="pt")
model.generate(**inputs, streamer=streamer)
```

### Pitfall 3: Not Handling Newlines

```python
# ❌ Bad: Newlines may not appear correctly
streamer = TextStreamer(tokenizer)

# ✅ Good: Ensure newlines are preserved
streamer = TextStreamer(
    tokenizer,
    decode_kwargs={"clean_up_tokenization_spaces": False}
)
```

---

## Summary

### Key Takeaways

1. **TextStreamer** enables real-time token-by-token output during generation
2. **Use it for** interactive apps, chatbots, long-form content, and debugging
3. **Configure with** `skip_prompt=True` and `skip_special_tokens=True` for clean output
4. **Performance cost** is minimal (~1-5% overhead)
5. **UX benefit** is significant - users see progress immediately

### Quick Reference

```python
# Standard setup for interactive applications
from transformers import TextStreamer

streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,              # Don't repeat input
    skip_special_tokens=True       # Clean output
)

# Use in generation
print("Response: ", end="", flush=True)
model.generate(
    **inputs,
    max_new_tokens=100,
    streamer=streamer,
    temperature=0.7,
    top_p=0.9
)
print()  # Newline after completion
```

### When to Use

- ✅ **Interactive chatbots** - Users see responses immediately
- ✅ **Long-form generation** - Users can read while generating
- ✅ **Debugging** - Monitor generation in real-time
- ✅ **Demos** - Show model capabilities dynamically
- ❌ **Batch processing** - No user waiting for output
- ❌ **API endpoints** - Return complete response
- ❌ **Offline processing** - No need for real-time feedback

---

## Further Reading

- [HuggingFace TextStreamer Documentation](https://huggingface.co/docs/transformers/main/en/internal/generation_utils#transformers.TextStreamer)
- [Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Streaming in Production](https://huggingface.co/docs/transformers/main/en/llm_tutorial)

## Related Tutorials

- [Model Loading Guide](../model-downloading-guide.md)
- [Usage Examples](../usage-examples.md)
- [Generation Parameters Guide](generation-parameters.md) *(coming soon)*
