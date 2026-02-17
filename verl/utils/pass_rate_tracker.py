"""
Standalone tracker for historical pass rates which can be used with multiple samplers.

    - Tracks success and attempt counts for each sample in the dataset
    - Can be used with different sampling strategies (e.g., weighted sampling based on pass rates), 
        see `PassRateWeightedSampler` for an example
"""

import numpy as np

class PassRateTracker:
    """
    Tracks pass rates for all samples in the dataset.
    Uses dataset indices (0, 1, 2, ..., N-1) as persistent sample IDs.
    
    This class only tracks pass rates; weighting and sampling strategies are implemented
    separately (see `PassRateWeightedSampler` for an example of how pass rates can be
    converted into sampling weights).
    """
    
    def __init__(self, dataset_size: int, use_ema: bool = False, ema_alpha: float = 0.1):
        """
        Args:
            dataset_size: Total number of samples in the dataset
            use_ema: If True, use exponential moving average for pass rates
            ema_alpha: EMA smoothing factor (0 to 1). Higher = more weight to recent updates
        """
        self.dataset_size = dataset_size
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        
        # Track stats for each sample index
        self.attempt_counts = np.zeros(dataset_size, dtype=np.int32)
        
        # Initialize pass_rate to -5.0 for all untried samples which forces the model to sample each prompt at least once
        # before curriculum based sampling starts
        # TODO (Jalaj): Consider changing this strategy to read and write from a file to persist across training runs rather than keeping pass rates in memory
        
        # Keep both pass_rate and ema_pass_rate to enable future analysis and comparison:
        # TODO (Jalaj): Detecting high-variance samples or sudden performance changes and do adaptive weighting strategies based on learning dynamics
        self.pass_rate = -5 * np.ones(dataset_size, dtype=np.float16)
        self.ema_pass_rate = -5 * np.ones(dataset_size, dtype=np.float16)
    
    def update(self, sample_indices: np.ndarray, batch_pass_rate: np.ndarray):
        """
        Update pass rate statistics for a batch of samples.
        
        Args:
            sample_indices: Array of dataset indices, shape (batch_size,)
            batch_pass_rate: Array indicating average pass rate for each sample, shape (batch_size,)
        """
        assert len(sample_indices) == len(batch_pass_rate), \
            f"Mismatch: {len(sample_indices)} indices vs {len(batch_pass_rate)} pass rates"
        
        # Increment attempt count, this can be used for sampling with bandit style algorithms
        self.attempt_counts[sample_indices] += 1
        
        # Update latest pass rate with this batch's result
        self.pass_rate[sample_indices] = batch_pass_rate
        
        # Update EMA pass rate if enabled
        if self.use_ema:
            old_ema = self.ema_pass_rate[sample_indices]    # Get current EMA values for the batch
            first_time_mask = old_ema < 0   # Create mask for first-time updates (negative values)
            
            # Compute new EMA values:
            # For first time: use batch_pass_rate directly
            # For subsequent: apply EMA formula
            new_ema = np.where(
                first_time_mask,
                batch_pass_rate,  # First time: just use current value
                self.ema_alpha * batch_pass_rate + (1 - self.ema_alpha) * old_ema  # EMA update
            )

            # Update all at once
            self.ema_pass_rate[sample_indices] = new_ema
    
    def get_pass_rates(self) -> np.ndarray:
        """
        Compute pass rates for all samples. For now, just use the historical pass rates (and optionally an
        exponential moving average); can be extended in many different ways
        
        Returns:
            Array of shape (dataset_size,) with pass rates in [0, 1] or [-5, 0] for untried.
            Uses EMA if self.use_ema=True, otherwise returns latest pass rates.
        """
        if self.use_ema:
            return self.ema_pass_rate
        else:
            return self.pass_rate

    def get_count_distribution_statistics(self) -> dict:
        """
        Return metrics for monitoring how count distribution changes over training steps

        Returns:
            Dict with percentiles of `attempt_counts` distribution
        """
        return {
            f'count_p{p}': float(np.percentile(self.attempt_counts, p)) 
            for p in range(5, 100, 5)
        }
    
    # For saving pass_rate_tracker state when saving training checkpoint
    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            'attempt_counts': self.attempt_counts.copy(),
            'pass_rate': self.pass_rate.copy(),
            'ema_pass_rate': self.ema_pass_rate.copy(),
        }
    
    # For loading pass_rate_tracker state when resuming training from checkpoint
    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.attempt_counts = state_dict['attempt_counts'].copy()
        self.pass_rate = state_dict['pass_rate'].copy()
        self.ema_pass_rate = state_dict['ema_pass_rate'].copy()