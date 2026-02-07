---
title: "On-Policy vs Off-Policy Reinforcement Learning"
author: "LLM Lab"
date: "December 2025"
---

# On-Policy vs Off-Policy Reinforcement Learning

This tutorial provides a clear comparison of **on-policy** and **off-policy** reinforcement learning, using **SARSA** and **Q-learning** as canonical examples. We'll build intuition first, then dive into the math, and finally connect these concepts to modern LLM training.

---

## 1. The Core Difference

### 1.1 On-Policy RL (e.g., SARSA)

**The policy you evaluate is the same policy you use to collect data.**

- The update target reflects the action *actually taken* under the current behavior policy
- The learned value function tracks the performance of the *actual* policy — including any exploration noise (e.g., ε-greedy randomness)
- Tends to be more conservative, because the value estimates account for the fact that the agent sometimes explores suboptimally

### 1.2 Off-Policy RL (e.g., Q-learning)

**The policy you evaluate (the *target* policy) differs from the policy you use to collect data (the *behavior* policy).**

- The update target uses a *different* action-selection rule than the one that generated the data — in Q-learning, the update always uses the greedy $\max$ action, even if the agent took a random exploratory action
- This means the learned value function reflects the performance of the *optimal* policy, not the exploratory behavior policy
- More sample-efficient in principle (can learn from any data source), but the mismatch between target and behavior policies can cause instability, especially with function approximation

> **Common misconception:** Using an ε-greedy behavior policy is *not* what makes Q-learning off-policy. SARSA can also use ε-greedy exploration. The distinction is in the **update rule**: Q-learning bootstraps from $\max_{a'} Q(s', a')$ (the target policy), while SARSA bootstraps from $Q(s', a')$ where $a'$ is the action actually taken (the behavior policy).

### 1.3 Quick Comparison

| Type | Update Target Uses | Learns Value Of | Behavior Policy |
|------|-------------------|-----------------|----------------|
| **On-policy (SARSA)** | Action actually taken ($a'$ from behavior policy) | The behavior policy (including exploration) | Same as target policy |
| **Off-policy (Q-learning)** | Best action ($\max_{a'} Q$, from target policy) | The optimal greedy policy | Can differ from target policy |

---

## 2. The Update Equations

Both SARSA and Q-learning are **temporal difference (TD)** methods that update Q-values based on the difference between predicted and observed returns.

### 2.1 SARSA (On-Policy)

SARSA updates using the *action actually taken* next:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

where:

| Symbol | Meaning |
|--------|---------|
| $Q(s, a)$ | Current Q-value for state $s$ and action $a$ |
| $\alpha$ | Learning rate |
| $r$ | Reward received after taking action $a$ in state $s$ |
| $\gamma$ | Discount factor |
| $s'$ | Next state |
| $a'$ | **Actual next action taken** (sampled from behavior policy) |

**Key point:** The update bootstraps from $Q(s', a')$, where $a'$ is the action *actually sampled* from the current behavior policy. This means SARSA's value estimates reflect the true performance of the policy being followed — including the cost of occasional random exploration.

### 2.2 Q-Learning (Off-Policy)

Q-learning updates using the *best possible action* next:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Key point:** The update bootstraps from $\max_{a'} Q(s', a')$ — the value of the *greedy* action — regardless of what action the agent actually took next. This is precisely what makes Q-learning off-policy: the **target policy** (greedy) differs from the **behavior policy** (e.g., ε-greedy). The agent learns the value of acting optimally, even while behaving exploratorily.

### 2.3 Side-by-Side Comparison

| Aspect | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|--------|-------------------|-------------------------|
| **Update target** | $r + \gamma Q(s', a')$ | $r + \gamma \max_{a'} Q(s', a')$ |
| **$a'$ in the target** | Sampled from behavior policy | Greedy (best action), regardless of what was taken |
| **Policy learned** | The behavior policy (with exploration) | The optimal greedy policy |
| **Why on/off-policy?** | Target and behavior policy are the same | Target policy (greedy) ≠ behavior policy (exploratory) |

---

## 3. A Concrete Example: The Cliff Walking Problem

This classic example from Sutton & Barto (2018, Example 6.6) makes the on/off-policy distinction viscerally clear.

**Setup:** An agent navigates a gridworld from Start to Goal. The bottom row contains a cliff — stepping on any cliff cell gives a large negative reward ($-100$) and resets the agent to Start. All other moves give $-1$. There are two routes: a **safe path** along the top (longer, but no risk) and an **optimal path** along the bottom edge (shorter, but one misstep sends you off the cliff).

### 3.1 SARSA (On-Policy) Behavior

SARSA learns the value of the policy it actually follows. With ε-greedy exploration, the agent occasionally takes random actions. Near the cliff edge, a random action can be fatal. SARSA's value estimates *incorporate* this risk:

> "Walking along the cliff edge is dangerous *for me*, because I sometimes explore randomly and fall off."

**Result:** SARSA learns to take the **safe path** — it stays far from the cliff because its value estimates honestly reflect the cost of its own exploration mistakes.

### 3.2 Q-Learning (Off-Policy) Behavior

Q-learning learns the value of the *optimal* (greedy) policy, regardless of the exploratory behavior. The greedy policy would never step off the cliff, so Q-learning's value estimates ignore the exploration risk:

> "The cliff-edge path is optimal *if I always act greedily* — the fact that I sometimes explore is irrelevant to my value estimates."

**Result:** Q-learning learns that the **optimal path** runs along the cliff edge. But during training, the agent actually *does* explore and repeatedly falls off the cliff, leading to worse online performance despite learning the theoretically optimal policy.

### 3.3 Why This Matters

| Algorithm | Learned Path | Online Performance | Reason |
|-----------|-------------|-------------------|--------|
| **SARSA** | Safe (away from cliff) | Better during training | Value estimates account for exploration risk |
| **Q-learning** | Optimal (along cliff edge) | Worse during training | Value estimates assume greedy behavior |

This illustrates a fundamental trade-off: on-policy methods learn *realistic* values (what will actually happen), while off-policy methods learn *idealized* values (what would happen under optimal behavior). Neither is universally better — it depends on whether you care about performance *during* learning or performance *after* learning (once exploration stops).

---

## 4. Connection to LLM Training (RLHF/RLAIF)

Modern LLM training with reinforcement learning has interesting connections to both on-policy and off-policy concepts. The picture is more nuanced than a simple classification.

### 4.1 The On-Policy Core

The most common RLHF algorithm, **PPO** (Proximal Policy Optimization), is fundamentally an **on-policy** method:

- Each training iteration generates **fresh rollouts** from the current policy
- The policy update uses these on-policy samples (with a clipped surrogate objective)
- Old rollouts are discarded after each update

In this sense, the inner loop of RLHF is on-policy — just like SARSA learns from its own behavior, PPO learns from the current policy's outputs.

### 4.2 The Off-Policy Complications

However, several aspects of the RLHF pipeline introduce **off-policy-like challenges**:

- **Reward model drift:** The reward model is trained on human preferences collected from *earlier* policy versions. As the policy improves, the reward model may become stale or miscalibrated for the new policy's outputs.
- **Distribution shift:** Even within a single PPO iteration, the policy changes over multiple gradient steps while the rollout data stays fixed — a mild form of off-policy learning.
- **Evaluation mismatch:** The policy is often evaluated using a *different* reward signal than the one used during training (e.g., a held-out reward model, human evaluators, or automated verifiers).

### 4.3 Implications for Evaluation

| Aspect | Implication |
|--------|-------------|
| **Reward model staleness** | The RM may not accurately score the improved policy's outputs |
| **Distribution shift** | Training data comes from a slightly older policy version |
| **Evaluation protocol** | Final evaluation should use a *separate* reward signal (fresh RM, verifiers, or human judges) to avoid overfitting to the training RM |

This last point mirrors **off-policy evaluation** in classical RL: we want to assess a target policy (the trained LLM) using a reward signal that is independent of the training process. Just as Q-learning evaluates the greedy policy using data from an exploratory policy, RLHF evaluation assesses the final policy using a reward model it wasn't directly optimized against.

---

## 5. Summary

### 5.1 Key Differences

| Aspect | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|--------|-------------------|-------------------------|
| **Learns value of** | Behavior policy | Optimal policy |
| **Updates with** | Actual next action $a'$ | Greedy next action $\max_{a'}$ |
| **Exploration impact** | Reflected in value estimates | Not reflected in value estimates |
| **Stability** | Generally more stable (no target–behavior mismatch) | Can diverge with function approximation (the "deadly triad") |
| **Risk behavior** | Risk-averse (accounts for exploration mistakes) | Risk-neutral w.r.t. exploration (assumes greedy execution) |
| **LLM RL (RLHF)** | PPO (the most common RLHF algorithm) is on-policy | Off-policy *concepts* apply to evaluation and reward model drift |

### 5.2 When to Use Which

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Safety-critical environments | On-policy | Value estimates reflect actual behavior, including exploration risk |
| Learning optimal behavior from diverse data | Off-policy | Can learn from replay buffers, demonstrations, or other policies |
| Experience replay / replay buffers | Off-policy | Data was collected by older policies |
| LLM fine-tuning (RLHF) | On-policy (PPO), with off-policy evaluation | Fresh rollouts for training; separate reward signal for evaluation |

---

## 6. Further Topics

- Visual diagrams comparing SARSA vs Q-learning update paths
- Python implementation of both algorithms on the cliff walking problem
- The "deadly triad" — why off-policy + function approximation + bootstrapping can diverge
- Connection to PPO, GRPO, DPO, and post-training in modern LLMs
- Importance sampling and its role in bridging on-policy and off-policy methods
- How off-policy evaluation techniques apply to RLHF evaluation protocols

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
2. Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards* (Q-learning)
3. Rummery, G. A., & Niranjan, M. (1994). *On-line Q-learning using connectionist systems* (SARSA)
