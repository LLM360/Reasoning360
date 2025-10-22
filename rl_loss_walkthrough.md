# RL Loss Computation Walkthrough

## Overview
This document walks through how the RL loss is computed in the training pipeline, starting from the training script.

## Call Stack

```
scripts/train/hft/qwen3_8b_fsdp_single_node.sh
  └─> recipe.dapo.main_dapo (line 178)
      └─> RayDAPOTrainer.fit() (recipe/dapo/dapo_ray_trainer.py)
          └─> self.actor_rollout_wg.update_actor(batch) (verl/trainer/ppo/ray_trainer.py:1138)
              └─> ActorRolloutRefWorker.update_actor(data) (verl/workers/fsdp_workers.py:819)
                  └─> self.actor.update_policy(data) (verl/workers/fsdp_workers.py:831)
                      └─> DataParallelPPOActor.update_policy(data) (verl/workers/actor/dp_actor.py:363)
```

## Detailed Walkthrough

### 1. Training Script Entry Point
**File**: `scripts/train/hft/qwen3_8b_fsdp_single_node.sh:178`

The script launches the DAPO trainer with all the hyperparameters:
- `adv_estimator=grpo` - Using GRPO (Group Relative Policy Optimization)
- `clip_ratio_low=0.2`, `clip_ratio_high=0.2` - PPO clipping parameters
- `loss_agg_mode="token-mean"` - How to aggregate losses across tokens
- Various other training configs (learning rate, batch sizes, etc.)

### 2. Trainer Main Loop
**File**: `verl/trainer/ppo/ray_trainer.py:1138`

Inside the training loop, after collecting rollouts and computing advantages:
```python
with marked_timer("update_actor", timing_raw, color="red"):
    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
    actor_output = self.actor_rollout_wg.update_actor(batch)
```

The `batch` contains:
- `responses` - Generated responses from the model
- `old_log_probs` - Log probabilities from the rollout
- `advantages` - Computed advantage values (rewards processed through advantage estimator)
- `response_mask` - Mask for valid response tokens
- Other metadata

### 3. Worker Update Actor
**File**: `verl/workers/fsdp_workers.py:819-858`

The worker handles model loading/offloading for memory efficiency, then calls the actor's update_policy:
```python
def update_actor(self, data: DataProto):
    assert self._is_actor
    if self._is_offload_param:
        load_fsdp_model_to_gpu(self.actor_module_fsdp)
    if self._is_offload_optimizer:
        load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

    with self.ulysses_sharding_manager:
        data = data.to("cpu")

        # CORE COMPUTATION HERE
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(data=data)  # ← RL LOSS COMPUTED HERE

        # Log metrics, handle memory, etc.
        ...

    return output
```

### 4. **THE CORE: RL Loss Computation**
**File**: `verl/workers/actor/dp_actor.py:363-499`

This is where the actual RL loss is computed. Let me break it down:

#### Step 1: Data Preparation (lines 363-395)
```python
def update_policy(self, data: DataProto):
    self.actor_module.train()

    # Extract required keys from the batch
    select_keys = [
        "responses",           # Generated text tokens
        "response_mask",       # Valid token mask
        "input_ids",           # Prompt tokens
        "attention_mask",      # Attention mask
        "position_ids",        # Position embeddings
        "old_log_probs",       # Log probs from rollout policy
        "advantages",          # Computed advantages from rewards
    ]
    if self.config.use_kl_loss:
        select_keys.append("ref_log_prob")  # Reference model log probs
```

#### Step 2: Mini-batch Loop (lines 395-411)
```python
    # Split to make minibatch iterator (standard PPO)
    mini_batches = data.split(self.config.ppo_mini_batch_size)

    metrics = {}
    for _ in range(self.config.ppo_epochs):  # Multiple epochs over same data
        for batch_idx, mini_batch in enumerate(mini_batches):
            # Prepare dynamic batches for memory efficiency
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()
```

#### Step 3: Micro-batch Forward Pass (lines 413-436)
```python
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                # Extract key tensors
                response_mask = model_inputs["response_mask"]      # (bsz, response_length)
                old_log_prob = model_inputs["old_log_probs"]       # (bsz, response_length)
                advantages = model_inputs["advantages"]            # (bsz, response_length)

                # Forward pass through current policy to get NEW log probs
                entropy, log_prob = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=True
                )
                # entropy: (bsz, response_length) - entropy of current policy
                # log_prob: (bsz, response_length) - log probs from current policy
```

#### Step 4: **POLICY GRADIENT LOSS** (lines 443-456)
```python
                # Get the policy loss function based on config
                loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                # Options: "vanilla" (standard PPO), "gpg", "clip_cov"
                policy_loss_fn = get_policy_loss_fn(loss_mode)

                # COMPUTE PPO CLIPPED LOSS
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                    old_log_prob=old_log_prob,      # From rollout
                    log_prob=log_prob,               # From current policy
                    advantages=advantages,            # Processed rewards
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,     # "token-mean"
                    config=self.config,
                    rollout_log_probs=rollout_log_probs,
                )
```

**What happens in `policy_loss_fn` (vanilla PPO)**:
```python
# Pseudo-code for vanilla PPO loss:
ratio = exp(log_prob - old_log_prob)  # π_θ(a|s) / π_θ_old(a|s)

# PPO clipped objective
clipped_ratio = clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
policy_loss = -min(ratio * advantages, clipped_ratio * advantages)

# Aggregate over tokens
pg_loss = mean(policy_loss * response_mask) / mean(response_mask)
```

#### Step 5: Add Entropy Bonus (lines 458-464)
```python
                if entropy_coeff != 0:
                    # Aggregate entropy across valid tokens
                    entropy_loss = agg_loss(
                        loss_mat=entropy,
                        loss_mask=response_mask,
                        loss_agg_mode=loss_agg_mode
                    )

                    # Add entropy to encourage exploration
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                else:
                    policy_loss = pg_loss
```

#### Step 6: Optional KL Divergence Loss (lines 466-476)
```python
                if self.config.use_kl_loss:
                    ref_log_prob = model_inputs["ref_log_prob"]

                    # Compute KL divergence from reference model
                    kld = kl_penalty(
                        logprob=log_prob,
                        ref_logprob=ref_log_prob,
                        kl_penalty=self.config.kl_loss_type
                    )
                    kl_loss = agg_loss(
                        loss_mat=kld,
                        loss_mask=response_mask,
                        loss_agg_mode=loss_agg_mode
                    )

                    # Add KL penalty to keep policy close to reference
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

#### Step 7: Backward Pass (lines 478-497)
```python
                # Scale loss for gradient accumulation
                if self.config.use_dynamic_bsz:
                    loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                else:
                    loss_scale_factor = 1 / self.gradient_accumulation

                loss = policy_loss * loss_scale_factor
                loss.backward()  # ← BACKPROP HAPPENS HERE

                # Collect metrics
                micro_batch_metrics.update({
                    "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                })

            # Update weights after accumulating all micro-batches
            grad_norm = self._optimizer_step()  # ← OPTIMIZER STEP HERE
```

## Final Loss Formula

The complete RL loss is:

```
Total Loss = PPO_Clipped_Loss - entropy_coeff * Entropy + kl_loss_coef * KL_Divergence

Where:
  PPO_Clipped_Loss = -E[ min(ratio * advantages, clip(ratio, 1-ε, 1+ε) * advantages) ]
  ratio = π_θ(a|s) / π_θ_old(a|s) = exp(log_prob - old_log_prob)
  Entropy = -Σ p(a|s) * log p(a|s)
  KL_Divergence = KL(π_θ || π_ref)
```

With your config:
- `clip_ratio_low = 0.2` → ε_low = 0.2
- `clip_ratio_high = 0.2` → ε_high = 0.2
- `entropy_coeff = 0` → No entropy bonus
- `use_kl_loss = False` → No KL penalty
- `loss_agg_mode = "token-mean"` → Average across tokens

So effectively:
```
Total Loss = -E[ min(ratio * advantages, clip(ratio, 0.8, 1.2) * advantages) ]
```

## Key Files for Modification

If you want to add SFT loss for dynamic SFT/RL switching, you should modify:

1. **Primary location**: `verl/workers/actor/dp_actor.py:363-499` (update_policy method)
   - This is where you'd add conditional logic to compute SFT loss instead of/alongside RL loss
   - You have access to all the data including `advantages`, `responses`, etc.
   - You can add teacher responses to the DataProto batch

2. **Data loading**: `verl/utils/dataset/rl_dataset.py`
   - Already loads teacher responses from parquet files
   - You could add logic to select best teacher response based on pass rate

3. **Advantage computation**: `verl/trainer/ppo/ray_trainer.py`
   - Where advantages are computed from rewards
   - You could use advantages/rewards to decide SFT vs RL mode

## Next Steps

For implementing dynamic SFT/RL:
1. Add a flag to the batch indicating whether to use SFT or RL mode
2. In `update_policy`, check this flag:
   - If SFT mode: compute cross-entropy loss against teacher response
   - If RL mode: use existing PPO loss
3. The flag can be set based on student pass_rate from the data
