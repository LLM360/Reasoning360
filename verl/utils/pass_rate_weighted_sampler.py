"""
Weighted Sampler for curriculum learning.

Uses PassRateTracker to compute dynamic sampling weights based on pass rates.
"""

import numpy as np
from omegaconf import DictConfig

from verl.experimental.dataset.sampler import AbstractSampler
from verl.utils.pass_rate_tracker import PassRateTracker


class PassRateWeightedSampler(AbstractSampler):
    """
    Weighted sampler that uses pass rates to adjust sampling probabilities.
    
    Implements curriculum learning by dynamically adjusting sampling weights
    based on per-sample success rates. Samples with lower pass rates are sampled more frequently.
    """
    
    def __init__(self, data_source, data_config: DictConfig):
        """
        Args:
            data_source: The dataset object (Sized)
            data_config: Configuration dictionary containing the entire data config
        """
        self.data_source = data_source
        self.data_config = data_config
        self.dataset_size = len(data_source)
        
        # Temperature parameter for controlling the sharpness of the weighting distribution (from sampler config)
        #  - temperature < 1.0: Sharp (hard samples dominate)
        #  - temperature = 1.0: Balanced
        #  - temperature > 1.0: Soft (nearly uniform)
        self.temperature = data_config.sampler.get("pass_rate_temperature", 1.0)
        use_ema = data_config.sampler.get("use_ema", False)
        ema_alpha = data_config.sampler.get("ema_alpha", 0.1)

        # Create tracker for this dataset: set `use_ema=True` for exponential moving average pass rates
        self.pass_rate_tracker = PassRateTracker(dataset_size=self.dataset_size, use_ema=use_ema, ema_alpha=ema_alpha)
        self._cached_weights = None  # Cache for weights to avoid recomputation each iteration
        self._last_pass_rate = None  # Track pass rate in the previous training step
        self._update_tolerance = 0.01 # Tolerance for detecting significant pass rate changes which require recomputing the weight vector
    
    def __len__(self):
        return self.dataset_size

    def get_weights(self) -> np.ndarray:
        """
        We can add different weighting strategies here. Todo (Jalaj): add an additional argument to select strategy
        
        Current strategy: compute sampling weights inversely proportional to pass rates.
        - Untried samples (pass_rate=-5.0): weight = exp(5.0/temperature) -- baseline
        - Tried but failing (pass_rate≈0): weight = exp(0/temperature) = 1.0 -- highest priority after trying atleast once
        - Tried and succeeding (pass_rate>0): weight = exp(-pass_rate/temperature) -- lower priority
        
        Returns:
            Array of shape (dataset_size,) with unnormalized sampling weights
        """
        pass_rates_current_train_step = self.pass_rate_tracker.get_pass_rates()
        # Check if we need to recompute (vectorized comparison)
        if self._cached_weights is None or self._last_pass_rate is None:
            needs_update = True
        else:
            max_change = np.abs(pass_rates_current_train_step - self._last_pass_rate).max()
            needs_update = max_change > self._update_tolerance
        
        if needs_update:
            # ------  Weight inversely proportional to pass rate ---------

            ## Option 1: weights = np.power(1.0 - pass_rates, 1.0 / max(temperature, 0.01))
            # weights = np.exp(1 - pass_rates / max(self.temperature, 0.01))
            ## Stable implementation using log-exp trick
            # log_weights = (1.0/max(temperature, 0.01)) * np.log(1.0 - pass_rates + 1e-10)  # log(1 - p)
            # weights = np.exp(log_weights - log_weights.max())  # stable softmax exp(-(y - max_y)/temperature)

            # Option 2: negative exponential scaling
            x = -pass_rates_current_train_step / max(self.temperature, 0.01)
            # weights = np.exp(x)
            self._cached_weights = np.exp(x - x.max())  # stable softmax exp(-(y - max_y)/temperature)
            self._last_pass_rate = pass_rates_current_train_step.copy()

        return self._cached_weights
    
    def __iter__(self):
        """
        Generate indices for one epoch using current pass rate weights.
        """
        # Get current weights from tracker
        weights = self.get_weights()
        
        # Sample with replacement using weights
        # TODO: change this to make it scalable for large datasets
        indices = np.random.choice(
            self.dataset_size,
            size=self.dataset_size,
            replace=True,
            p=weights / weights.sum()  # Normalize to probability
        )
        
        return iter(indices)

    def get_weight_distribution_statistics(self) -> dict:
        """
        Return metrics for monitoring how weight distribution change over training steps

        Returns:
            Dict with percentiles of weight distributions
        """
        weights = self.get_weights()
        return {
            f'weight_p{p}': float(np.percentile(weights, p)) 
            for p in range(5, 100, 5)
        }

    def get_wandb_3d_plot_data(self, metric_type: str = 'weight') -> list:
        """
        Prepare data for W&B 3D plot: percentiles (x), values (y), step (z).
        
        Args:
            metric_type: 'weight' or 'count'
            
        Returns:
            List of dicts with percentile, value for 3D plotting
        """
        if metric_type == 'weight':
            stats = self.get_weight_distribution_statistics()
        elif metric_type == 'count':
            stats = self.pass_rate_tracker.get_count_distribution_statistics() # get count stats from tracker
        else:
            raise ValueError(f"metric_type must be 'weight' or 'count', got {metric_type}")
        
        # Build data list directly from percentiles
        return [
            {
                'percentile': p,
                'percentile_name': f'p{p}',
                'value': stats[f'{metric_type}_p{p}'],
            }
            for p in range(5, 100, 5)
        ]

    def state_dict(self) -> dict:
        """
        Return state for checkpointing.
        Includes the pass rate tracker state so it can be restored on resume.
        """
        return self.pass_rate_tracker.state_dict()
    
    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load state from checkpoint.
        Restores the pass rate tracker state.
        """
        self.pass_rate_tracker.load_state_dict(state_dict)
