# Direct Preference Optimization (DPO): A Tutorial

This tutorial provides a comprehensive introduction to **Direct Preference Optimization (DPO)**, a breakthrough technique for aligning large language models with human preferences. We will build up the intuition step by step, walk through the mathematics carefully, and connect everything back to practical implementation.

By the end of this tutorial, you will understand:

- Why DPO was developed and what problem it solves
- How DPO relates to (and improves upon) RLHF
- The mathematical derivation and intuition behind the DPO loss
- How to interpret each component of the loss function
- Practical considerations for training with DPO

---

## Table of Contents

1. [The Alignment Problem](#1-the-alignment-problem)
2. [From RLHF to DPO: A Tale of Two Approaches](#2-from-rlhf-to-dpo-a-tale-of-two-approaches)
3. [Understanding the DPO Loss Function](#3-understanding-the-dpo-loss-function)
4. [Step-by-Step Walkthrough](#4-step-by-step-walkthrough)
5. [The Role of the Reference Model](#5-the-role-of-the-reference-model)
6. [The β Hyperparameter](#6-the-β-hyperparameter)
7. [DPO as Energy-Based Modeling](#7-dpo-as-energy-based-modeling)
8. [Mathematical Derivation from RLHF](#8-mathematical-derivation-from-rlhf)
9. [Implementing DPO in Code](#9-implementing-dpo-in-code)
10. [Why DPO Works Better Than PPO](#10-why-dpo-works-better-than-ppo)
11. [Notation Reference](#11-notation-reference)

---

## 1. The Alignment Problem

Large language models are trained on vast amounts of internet text, which means they learn to predict what text *typically* comes next—not necessarily what text *should* come next according to human values. A model might generate text that is factually incorrect, harmful, biased, or simply unhelpful, because such text exists in its training data.

**Alignment** is the process of adjusting a model's behavior so that it produces outputs that humans actually prefer. But how do we teach a model what humans prefer?

The most direct approach is to collect **preference data**: given a prompt, show humans two different responses and ask them which one they prefer. This gives us triplets of the form:

- **x**: the input prompt
- **y₊** (y-plus): the preferred response (chosen by humans)
- **y₋** (y-minus): the dispreferred response (rejected by humans)

The question then becomes: how do we use these preference pairs to train the model?

---

## 2. From RLHF to DPO: A Tale of Two Approaches

### The RLHF Approach

Before DPO, the dominant approach was **Reinforcement Learning from Human Feedback (RLHF)**. RLHF works in three stages:

1. **Train a reward model**: Take the preference data and train a separate neural network R(x, y) that predicts how much humans would like response y given prompt x.

2. **Optimize the policy with RL**: Use reinforcement learning (typically PPO—Proximal Policy Optimization) to fine-tune the language model so that it generates responses with high reward.

3. **Add KL regularization**: To prevent the model from drifting too far from its original behavior (which could lead to reward hacking or degenerate outputs), add a penalty term that keeps the fine-tuned model close to the original.

While RLHF works, it has significant drawbacks:

- **Complexity**: You need to train and maintain two models (reward model + policy)
- **Instability**: PPO is notoriously finicky and requires careful hyperparameter tuning
- **Computational cost**: RL training requires generating rollouts, computing advantages, and multiple optimization steps
- **Reward hacking**: The policy can learn to exploit quirks in the reward model rather than genuinely improving

### The DPO Insight

In 2023, Rafailov et al. published a paper titled *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"* that changed everything. They made a remarkable observation:

> If we assume the optimal RLHF solution follows a specific mathematical form (the entropy-regularized policy optimum), then we can **analytically solve for the reward function in terms of the policy itself**.

This means we don't need a separate reward model at all. We can reparameterize the reward in terms of the language model's own probabilities and train directly on the preference data using a simple supervised loss.

DPO is often summarized as:

> **"RLHF without the RL."**

It achieves the same objective as RLHF—maximizing reward while staying close to the reference model—but through a purely supervised training procedure with no rollouts, no value functions, and no PPO.

---

## 3. Understanding the DPO Loss Function

The DPO loss function looks intimidating at first glance, but it has a beautiful intuitive interpretation. Let's build it up piece by piece.

### The Full Loss Function

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left( \beta \left[ \log \frac{\pi_\theta(y_+|x)}{\pi_{\text{ref}}(y_+|x)} - \log \frac{\pi_\theta(y_-|x)}{\pi_{\text{ref}}(y_-|x)} \right] \right)
$$

Where:
- $\pi_\theta$ is the model we're training (the "policy")
- $\pi_{\text{ref}}$ is a frozen copy of the original model (the "reference")
- $\sigma$ is the sigmoid function
- $\beta$ is a temperature hyperparameter

### Breaking It Down

Let's define some intermediate quantities to make this clearer.

**The "advantage" of the preferred response:**

$$
A_+ = \log \frac{\pi_\theta(y_+|x)}{\pi_{\text{ref}}(y_+|x)}
$$

This measures how much *more* (or less) the current model likes the preferred response compared to the reference model. If $A_+ > 0$, our training has made the model more likely to generate the good response.

**The "advantage" of the dispreferred response:**

$$
A_- = \log \frac{\pi_\theta(y_-|x)}{\pi_{\text{ref}}(y_-|x)}
$$

Similarly, this measures how the model's preference for the bad response has changed. Ideally, we want $A_- < 0$, meaning training has made the model *less* likely to generate the bad response.

**The preference margin:**

$$
D = A_+ - A_-
$$

This is the key quantity. It measures the *relative* change in preferences. A large positive $D$ means the model has learned to strongly prefer the good response over the bad one (relative to where it started).

**The final loss:**

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma(\beta \cdot D)
$$

This is simply **binary cross-entropy** applied to the preference margin. The sigmoid converts $D$ into a probability (the probability that the model prefers $y_+$ over $y_-$), and the negative log penalizes the model when this probability is low.

---

## 4. Step-by-Step Walkthrough

Let's walk through what happens during one training step with a concrete mental model.

### Step 1: Start with a preference pair

You have:
- A prompt: *"What is the capital of France?"*
- A preferred response: *"The capital of France is Paris."*
- A dispreferred response: *"France is a country in Europe."*

### Step 2: Compute log-probabilities

For both the training model ($\pi_\theta$) and the frozen reference model ($\pi_{\text{ref}}$), compute the log-probability of generating each response given the prompt.

Think of this as asking each model: "How likely would you be to generate this exact text?"

### Step 3: Compute the advantages

Calculate how much the training model's preferences have shifted relative to the reference:

$$
A_+ = \log \pi_\theta(y_+|x) - \log \pi_{\text{ref}}(y_+|x)
$$

$$
A_- = \log \pi_\theta(y_-|x) - \log \pi_{\text{ref}}(y_-|x)
$$

### Step 4: Compute the preference margin

$$
D = A_+ - A_-
$$

**Interpretation:**
- If $D > 0$: The model prefers the good response more than the bad one (relative to the reference). Good!
- If $D < 0$: The model prefers the bad response more. Bad!
- If $D = 0$: The model hasn't learned to distinguish them.

### Step 5: Apply the loss

$$
\mathcal{L} = -\log \sigma(\beta \cdot D)
$$

**What this does:**
- If $D$ is large and positive → $\sigma(\beta D) \approx 1$ → $\mathcal{L} \approx 0$ (no penalty)
- If $D$ is negative → $\sigma(\beta D) \approx 0$ → $\mathcal{L}$ is large (strong penalty)

The gradient will push the model to increase $D$—that is, to increase its relative preference for the good response.

---

## 5. The Role of the Reference Model

You might wonder: why do we need a reference model at all? Why not just maximize $\log \pi_\theta(y_+|x)$ and minimize $\log \pi_\theta(y_-|x)$?

The reference model serves a crucial purpose: **it prevents the model from drifting too far from its original capabilities**.

Without the reference model, the training process might:
- Collapse to always outputting the same "safe" response
- Lose its general language understanding
- Overfit to the specific phrasing in the preference data

By measuring changes *relative to the reference*, DPO implicitly includes a KL-divergence penalty that keeps the model close to its starting point. This is mathematically equivalent to the KL regularization term in RLHF, but it emerges naturally from the loss formulation rather than being added as a separate term.

Think of it this way: the reference model represents "what the model knew before alignment." We want to adjust its preferences while preserving its knowledge.

---

## 6. The β Hyperparameter

The $\beta$ parameter controls the "temperature" of the preference learning:

| β Value | Effect |
|---------|--------|
| **Small β (e.g., 0.1)** | Gentle preference learning. The model makes small adjustments and stays very close to the reference. Good for subtle alignment or when you want to preserve most of the original behavior. |
| **Medium β (e.g., 0.1–0.5)** | Balanced preference learning. This is the typical range used in practice. |
| **Large β (e.g., 1.0+)** | Aggressive preference learning. The model strongly enforces preferences but risks overfitting or instability. |
| **β = 0** | No preference learning at all (the loss becomes constant). |

In the RLHF framework, $\beta$ corresponds to the inverse of the KL regularization coefficient. A larger $\beta$ means less regularization, allowing the model to deviate more from the reference.

**Practical guidance:** Start with $\beta = 0.1$ and adjust based on your results. If the model isn't learning preferences strongly enough, increase $\beta$. If it's becoming unstable or losing capabilities, decrease it.

---

## 7. DPO as Energy-Based Modeling

There's an elegant way to view DPO through the lens of **energy-based models (EBMs)**.

Define the "energy" of a response as:

$$
E_\theta(x, y) = -\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

Lower energy means the model prefers that response more (relative to the reference).

The DPO objective then becomes:

> Push the energy of preferred responses **lower** than the energy of dispreferred responses.

This is exactly what contrastive learning does in other domains (like image representation learning). DPO is essentially **pairwise contrastive learning on language model outputs**.

The loss function:

$$
\mathcal{L} = -\log \sigma(\beta (E_-(x, y_-) - E_+(x, y_+)))
$$

penalizes the model when the bad response has lower energy (is more preferred) than the good response.

---

## 8. Mathematical Derivation from RLHF

For those interested in the mathematical foundations, here's how DPO is derived from the RLHF objective.

### The RLHF Objective

In RLHF, we want to maximize expected reward while staying close to the reference model:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

### The Optimal Policy

It can be shown that the optimal policy for this objective has the form:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

where $Z(x)$ is a normalizing constant.

### Reparameterizing the Reward

Solving for the reward in terms of the policy:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

### The Key Insight

When we compute the *difference* in rewards between two responses, the $Z(x)$ term cancels:

$$
r(x, y_+) - r(x, y_-) = \beta \left[ \log \frac{\pi^*(y_+|x)}{\pi_{\text{ref}}(y_+|x)} - \log \frac{\pi^*(y_-|x)}{\pi_{\text{ref}}(y_-|x)} \right]
$$

### From Reward Difference to Loss

The Bradley-Terry model for preferences says:

$$
P(y_+ \succ y_- | x) = \sigma(r(x, y_+) - r(x, y_-))
$$

Substituting our reparameterized reward and maximizing the likelihood of observed preferences gives us the DPO loss.

---

## 9. Implementing DPO in Code

Here's a simplified implementation of the DPO loss in PyTorch:

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log π_θ(y_+|x)
    policy_rejected_logps: torch.Tensor,  # log π_θ(y_-|x)
    ref_chosen_logps: torch.Tensor,       # log π_ref(y_+|x)
    ref_rejected_logps: torch.Tensor,     # log π_ref(y_-|x)
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute the DPO loss for a batch of preference pairs.
    
    Args:
        policy_chosen_logps: Log-probs of chosen responses under the policy
        policy_rejected_logps: Log-probs of rejected responses under the policy
        ref_chosen_logps: Log-probs of chosen responses under reference model
        ref_rejected_logps: Log-probs of rejected responses under reference model
        beta: Temperature parameter
    
    Returns:
        Scalar loss value
    """
    # Compute advantages (log-ratios)
    chosen_advantages = policy_chosen_logps - ref_chosen_logps
    rejected_advantages = policy_rejected_logps - ref_rejected_logps
    
    # Compute preference margin
    logits = beta * (chosen_advantages - rejected_advantages)
    
    # Binary cross-entropy loss (we want the "chosen" class to win)
    # This is equivalent to -log(sigmoid(logits))
    loss = -F.logsigmoid(logits).mean()
    
    return loss
```

In practice, you would use the `DPOTrainer` from the `trl` library, which handles all the details of computing log-probabilities, managing the reference model, and batching:

```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # Will create a copy automatically
    args=config,
    processing_class=tokenizer,
    train_dataset=preference_dataset,
)

trainer.train()
```

---

## 10. Why DPO Works Better Than PPO

DPO has several advantages over PPO-based RLHF:

### Stability

PPO requires careful tuning of many hyperparameters (clip ratio, value function coefficient, entropy bonus, etc.) and can be unstable during training. DPO is a simple supervised loss that behaves predictably.

### Simplicity

DPO eliminates the need for:
- A separate reward model
- Value function estimation
- Rollout generation
- Complex RL infrastructure

You just need preference data and a standard training loop.

### Computational Efficiency

PPO requires generating responses during training (rollouts), which is expensive. DPO works directly on pre-collected preference data, making it much faster.

### No Reward Hacking

Since there's no explicit reward model, the policy can't learn to exploit it. The "reward" is implicitly defined by the preference data.

### Theoretical Grounding

DPO is mathematically equivalent to RLHF under certain assumptions, so it achieves the same objective with fewer moving parts.

---

## 11. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x$ | Input prompt |
| $y_+$ | Preferred (chosen) response |
| $y_-$ | Dispreferred (rejected) response |
| $\pi_\theta(y \mid x)$ | Probability of the fine-tuned model generating response $y$ given prompt $x$ |
| $\pi_{\text{ref}}(y \mid x)$ | Probability of the frozen reference model generating response $y$ |
| $A_+ = \log \frac{\pi_\theta(y_+ \mid x)}{\pi_{\text{ref}}(y_+ \mid x)}$ | "Advantage" of the preferred response—how much more the model likes it compared to the reference |
| $A_- = \log \frac{\pi_\theta(y_- \mid x)}{\pi_{\text{ref}}(y_- \mid x)}$ | "Advantage" of the dispreferred response |
| $D = A_+ - A_-$ | Preference margin (reparameterized reward difference) |
| $\beta$ | Temperature hyperparameter controlling preference strength |
| $\sigma(\cdot)$ | Sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$ |
| $\mathcal{L}_{\text{DPO}}$ | The DPO loss function |

---

## Summary

DPO represents a significant simplification of the alignment pipeline. Instead of the complex RLHF setup with reward models and PPO, DPO shows that you can achieve the same goal with a simple supervised loss on preference pairs.

The key insights are:

1. **The reward is implicit in the policy.** By comparing the model's probabilities to a reference model, we can compute what the "reward" would be without ever training a reward model.

2. **Preference learning is contrastive learning.** DPO is essentially logistic regression on model log-probabilities, pushing the model to prefer good responses over bad ones.

3. **The reference model provides regularization.** By measuring changes relative to the reference, DPO naturally prevents the model from drifting too far from its original behavior.

4. **Simplicity enables scale.** Because DPO is just supervised learning, it's easy to implement, debug, and scale.

---

## Further Reading

- **Original DPO Paper**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- **RLHF Background**: Ouyang et al., "Training language models to follow instructions with human feedback" (2022)
- **DPO Variants**: IPO (Identity Preference Optimization), KTO (Kahneman-Tversky Optimization), ORPO (Odds Ratio Preference Optimization)

---

*This tutorial accompanies the notebook `posttrain_llm/L5/Lesson_5.ipynb`, which demonstrates DPO training in practice.*
