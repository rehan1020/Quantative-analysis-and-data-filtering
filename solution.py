import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass, field
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from itertools import combinations

warnings.filterwarnings('ignore')


@dataclass
class QuantizationResult:
    """Container for quantization results"""
    bucket_boundaries: List[float]
    bucket_labels: List[str]
    bucket_stats: pd.DataFrame
    optimization_method: str
    objective_value: float
    n_buckets: int
    
    def get_rating(self, fico_score: float) -> Optional[str]:
        """Map a FICO score to its rating bucket"""
        for i, (lower, upper) in enumerate(zip(
            [300] + self.bucket_boundaries[:-1], 
            self.bucket_boundaries
        )):
            if lower <= fico_score <= upper:
                return self.bucket_labels[i]
        return None
    
    def get_rating_numeric(self, fico_score: float) -> Optional[int]:
        """Map a FICO score to numeric rating (lower = better)"""
        rating = self.get_rating(fico_score)
        if rating is None:
            return None
        return int(rating.replace('R', ''))


class FICOQuantizer:
    """
    Optimal FICO score quantization using MSE or Log-Likelihood optimization.
    
    Maps continuous FICO scores (300-850) to categorical ratings where
    lower rating numbers indicate better credit quality.
    """
    
    MIN_FICO = 300
    MAX_FICO = 850
    
    def __init__(self, data: pd.DataFrame = None, 
                 fico_column: str = 'fico_score',
                 default_column: str = 'default'):
        """
        Initialize the quantizer.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing FICO scores and default labels
        fico_column : str
            Column name for FICO scores
        default_column : str
            Column name for default indicator (0/1)
        """
        self.fico_column = fico_column
        self.default_column = default_column
        self.data = data
        self.fico_values = None
        self.default_labels = None
        self.result = None
        
        if data is not None:
            self._load_data(data)
    
    def _load_data(self, data: pd.DataFrame):
        """Load and validate input data"""
        if self.fico_column not in data.columns:
            raise ValueError(f"Column '{self.fico_column}' not found in data")
        if self.default_column not in data.columns:
            raise ValueError(f"Column '{self.default_column}' not found in data")
        
        # Extract and clean data
        clean_data = data[[self.fico_column, self.default_column]].dropna()
        self.fico_values = clean_data[self.fico_column].values
        self.default_labels = clean_data[self.default_column].values.astype(int)
        
        # Validate FICO range
        if self.fico_values.min() < self.MIN_FICO or self.fico_values.max() > self.MAX_FICO:
            warnings.warn(f"FICO values outside expected range [{self.MIN_FICO}, {self.MAX_FICO}]")
    
    def _calculate_bucket_mse(self, fico_subset: np.ndarray, 
                             default_subset: np.ndarray) -> float:
        """
        Calculate Mean Squared Error for a single bucket.
        
        MSE = mean((fico_i - mean_fico)^2) for all i in bucket
        """
        if len(fico_subset) == 0:
            return np.inf
        mean_fico = np.mean(fico_subset)
        return np.mean((fico_subset - mean_fico) ** 2)
    
    def _calculate_bucket_log_likelihood(self, fico_subset: np.ndarray,
                                         default_subset: np.ndarray) -> float:
        """
        Calculate log-likelihood for a single bucket.
        
        LL = n * [p*log(p) + (1-p)*log(1-p)] where p = defaults/n
        This measures the information content of the bucket's default rate.
        """
        n = len(default_subset)
        if n == 0:
            return -np.inf
        
        k = np.sum(default_subset)
        p = k / n
        
        # Handle edge cases for log(0)
        if p == 0 or p == 1:
            return 0  # No information gain from certain outcomes
        
        # Log-likelihood of Bernoulli observations
        ll = n * (p * np.log(p) + (1 - p) * np.log(1 - p))
        return ll
    
    def _calculate_total_mse(self, boundaries: List[float], 
                           fico_sorted: np.ndarray,
                           default_sorted: np.ndarray) -> float:
        """Calculate total MSE across all buckets defined by boundaries"""
        total_mse = 0
        n_buckets = len(boundaries) + 1
        
        # Add sentinel boundaries
        all_boundaries = [self.MIN_FICO] + boundaries + [self.MAX_FICO + 1]
        
        for i in range(n_buckets):
            lower = all_boundaries[i]
            upper = all_boundaries[i + 1]
            
            # Find indices in this bucket
            mask = (fico_sorted >= lower) & (fico_sorted < upper)
            if i == n_buckets - 1:  # Last bucket includes upper bound
                mask = (fico_sorted >= lower) & (fico_sorted <= upper - 1)
            
            fico_bucket = fico_sorted[mask]
            default_bucket = default_sorted[mask]
            
            if len(fico_bucket) > 0:
                total_mse += len(fico_bucket) * self._calculate_bucket_mse(
                    fico_bucket, default_bucket)
        
        return total_mse
    
    def _calculate_total_log_likelihood(self, boundaries: List[float],
                                       fico_sorted: np.ndarray,
                                       default_sorted: np.ndarray) -> float:
        """Calculate total log-likelihood across all buckets"""
        total_ll = 0
        n_buckets = len(boundaries) + 1
        
        all_boundaries = [self.MIN_FICO] + boundaries + [self.MAX_FICO + 1]
        
        for i in range(n_buckets):
            lower = all_boundaries[i]
            upper = all_boundaries[i + 1]
            
            mask = (fico_sorted >= lower) & (fico_sorted < upper)
            if i == n_buckets - 1:
                mask = (fico_sorted >= lower) & (fico_sorted <= upper - 1)
            
            fico_bucket = fico_sorted[mask]
            default_bucket = default_sorted[mask]
            
            if len(fico_bucket) > 0:
                total_ll += self._calculate_bucket_log_likelihood(
                    fico_bucket, default_bucket)
        
        return total_ll
    
    def _optimize_boundaries_dp(self, n_buckets: int, method: str) -> List[float]:
        """
        Find optimal bucket boundaries using dynamic programming.
        
        DP formulation:
        - State: dp[i][j] = best objective value for first i unique FICO values into j buckets
        - Transition: dp[i][j] = min/max over k<j of {dp[k][j-1] + objective(k+1, i)}
        """
        # Get unique sorted FICO values as potential split points
        unique_fico = np.unique(self.fico_values)
        unique_fico = unique_fico[(unique_fico >= self.MIN_FICO) & 
                                  (unique_fico <= self.MAX_FICO)]
        
        if len(unique_fico) < n_buckets:
            warnings.warn(f"Only {len(unique_fico)} unique FICO values, "
                         f"reducing buckets to {len(unique_fico)}")
            n_buckets = len(unique_fico)
        
        # Sort data by FICO for efficient bucketing
        sort_idx = np.argsort(self.fico_values)
        fico_sorted = self.fico_values[sort_idx]
        default_sorted = self.default_labels[sort_idx]
        
        # Precompute objective values for all possible intervals
        n_unique = len(unique_fico)
        
        # Create index mapping for faster lookup
        fico_to_idx = {fico: idx for idx, fico in enumerate(unique_fico)}
        
        # Precompute cumulative sums for efficient interval calculations
        cumsum_fico = np.cumsum(np.insert(fico_sorted, 0, 0))
        cumsum_fico_sq = np.cumsum(np.insert(fico_sorted**2, 0, 0))
        cumsum_default = np.cumsum(np.insert(default_sorted, 0, 0))
        cumsum_count = np.arange(len(fico_sorted) + 1)
        
        def get_interval_stats(start_idx: int, end_idx: int):
            """Get stats for interval [start_idx, end_idx) in sorted data"""
            n = cumsum_count[end_idx] - cumsum_count[start_idx]
            if n == 0:
                return None, None, None, 0
            
            sum_fico = cumsum_fico[end_idx] - cumsum_fico[start_idx]
            sum_fico_sq = cumsum_fico_sq[end_idx] - cumsum_fico_sq[start_idx]
            sum_default = cumsum_default[end_idx] - cumsum_default[start_idx]
            
            mean_fico = sum_fico / n
            mse = (sum_fico_sq / n) - mean_fico**2
            p_default = sum_default / n if n > 0 else 0
            
            return mean_fico, mse, p_default, n
        
        def interval_objective(start_fico: float, end_fico: float) -> float:
            """Calculate objective for interval [start_fico, end_fico]"""
            mask = (fico_sorted >= start_fico) & (fico_sorted <= end_fico)
            fico_interval = fico_sorted[mask]
            default_interval = default_sorted[mask]
            
            if len(fico_interval) == 0:
                return np.inf if method == 'mse' else -np.inf
            
            if method == 'mse':
                return len(fico_interval) * self._calculate_bucket_mse(
                    fico_interval, default_interval)
            else:  # log_likelihood
                return self._calculate_bucket_log_likelihood(
                    fico_interval, default_interval)
        
        # DP tables
        # dp[i][j] = best objective for partitioning unique_fico[0:i] into j buckets
        INF = np.inf if method == 'mse' else -np.inf
        dp = np.full((n_unique + 1, n_buckets + 1), INF)
        parent = np.full((n_unique + 1, n_buckets + 1), -1)
        
        # Base case: 0 values, 0 buckets
        dp[0][0] = 0 if method == 'mse' else 0
        
        # Fill DP table
        for j in range(1, n_buckets + 1):  # Number of buckets
            for i in range(j, n_unique + 1):  # Number of unique FICO values
                best_val = INF
                best_k = -1
                
                # Try all possible positions for the last bucket boundary
                for k in range(j - 1, i):
                    if dp[k][j - 1] == INF:
                        continue
                    
                    # Calculate objective for bucket [k, i)
                    if k < n_unique and i <= n_unique:
                        start_fico = unique_fico[k] if k < n_unique else self.MIN_FICO
                        end_fico = unique_fico[i - 1] if i > 0 else self.MAX_FICO
                        
                        bucket_obj = interval_objective(start_fico, end_fico)
                        
                        if bucket_obj == (np.inf if method == 'mse' else -np.inf):
                            continue
                        
                        if method == 'mse':
                            candidate = dp[k][j - 1] + bucket_obj
                            if candidate < best_val:
                                best_val = candidate
                                best_k = k
                        else:  # log_likelihood (maximize)
                            candidate = dp[k][j - 1] + bucket_obj
                            if candidate > best_val:
                                best_val = candidate
                                best_k = k
                
                dp[i][j] = best_val
                parent[i][j] = best_k
        
        # Backtrack to find boundaries
        boundaries = []
        current_i = n_unique
        for j in range(n_buckets, 0, -1):
            k = parent[current_i][j]
            if k > 0 and k < n_unique:
                boundaries.append(unique_fico[k])
            current_i = k
        
        boundaries.reverse()
        return boundaries
    
    def _optimize_boundaries_greedy(self, n_buckets: int, method: str) -> List[float]:
        """
        Greedy approximation for boundary optimization (faster, near-optimal).
        
        Uses iterative splitting based on maximum objective improvement.
        """
        # Start with equal-width buckets as initial guess
        boundaries = [
            self.MIN_FICO + i * (self.MAX_FICO - self.MIN_FICO) / n_buckets
            for i in range(1, n_buckets)
        ]
        
        # Sort data
        sort_idx = np.argsort(self.fico_values)
        fico_sorted = self.fico_values[sort_idx]
        default_sorted = self.default_labels[sort_idx]
        
        # Iterative refinement
        max_iterations = 100
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(len(boundaries)):
                best_boundary = boundaries[i]
                best_objective = (np.inf if method == 'mse' else -np.inf)
                
                # Search neighborhood of current boundary
                search_range = np.linspace(
                    max(self.MIN_FICO, boundaries[i-1] + 1) if i > 0 else self.MIN_FICO,
                    min(self.MAX_FICO, boundaries[i+1] - 1) if i < len(boundaries)-1 else self.MAX_FICO,
                    50
                )
                
                for candidate in search_range:
                    test_boundaries = boundaries.copy()
                    test_boundaries[i] = candidate
                    
                    if method == 'mse':
                        obj = self._calculate_total_mse(
                            test_boundaries, fico_sorted, default_sorted)
                        if obj < best_objective:
                            best_objective = obj
                            best_boundary = candidate
                            improved = True
                    else:
                        obj = self._calculate_total_log_likelihood(
                            test_boundaries, fico_sorted, default_sorted)
                        if obj > best_objective:
                            best_objective = obj
                            best_boundary = candidate
                            improved = True
                
                boundaries[i] = best_boundary
            
            if not improved:
                break
        
        return sorted(boundaries)
    
    def quantize(self, n_buckets: int = 5, method: str = 'mse',
                use_dp: bool = True) -> QuantizationResult:
        """
        Perform FICO score quantization.
        
        Parameters:
        -----------
        n_buckets : int
            Number of rating buckets to create
        method : str
            Optimization method: 'mse' or 'log_likelihood'
        use_dp : bool
            Use dynamic programming (True) or greedy approximation (False)
        
        Returns:
        --------
        QuantizationResult with bucket boundaries and mapping functions
        """
        if self.fico_values is None:
            raise ValueError("No data loaded. Provide data during initialization or call _load_data()")
        
        if method not in ['mse', 'log_likelihood']:
            raise ValueError("method must be 'mse' or 'log_likelihood'")
        
        if n_buckets < 2 or n_buckets > 20:
            raise ValueError("n_buckets must be between 2 and 20")
        
        # Optimize boundaries
        if use_dp and len(self.fico_values) <= 10000:
            boundaries = self._optimize_boundaries_dp(n_buckets, method)
        else:
            boundaries = self._optimize_boundaries_greedy(n_buckets, method)
        
        # Ensure boundaries are within valid range and sorted
        boundaries = [max(self.MIN_FICO, min(self.MAX_FICO, b)) for b in boundaries]
        boundaries = sorted(list(set(boundaries)))
        
        # Adjust if we have fewer boundaries than expected
        while len(boundaries) < n_buckets - 1:
            # Add midpoint of largest gap
            all_points = [self.MIN_FICO] + boundaries + [self.MAX_FICO]
            max_gap = 0
            max_gap_idx = 0
            for i in range(len(all_points) - 1):
                gap = all_points[i + 1] - all_points[i]
                if gap > max_gap:
                    max_gap = gap
                    max_gap_idx = i
            new_boundary = (all_points[max_gap_idx] + all_points[max_gap_idx + 1]) / 2
            boundaries.append(int(new_boundary))
            boundaries = sorted(boundaries)
        
        # Calculate bucket statistics
        bucket_stats = self._calculate_bucket_statistics(boundaries)
        
        # Create rating labels (lower number = better credit)
        # R1 = best credit (highest FICO), R{n} = worst credit (lowest FICO)
        bucket_labels = [f"R{i+1}" for i in range(len(boundaries) + 1)]
        # Reverse so R1 corresponds to highest FICO bucket
        bucket_labels = bucket_labels[::-1]
        
        # Calculate objective value
        sort_idx = np.argsort(self.fico_values)
        fico_sorted = self.fico_values[sort_idx]
        default_sorted = self.default_labels[sort_idx]
        
        if method == 'mse':
            objective_value = self._calculate_total_mse(boundaries, fico_sorted, default_sorted)
        else:
            objective_value = self._calculate_total_log_likelihood(boundaries, fico_sorted, default_sorted)
        
        self.result = QuantizationResult(
            bucket_boundaries=boundaries,
            bucket_labels=bucket_labels,
            bucket_stats=bucket_stats,
            optimization_method=method,
            objective_value=objective_value,
            n_buckets=n_buckets
        )
        
        return self.result
    
    def _calculate_bucket_statistics(self, boundaries: List[float]) -> pd.DataFrame:
        """Calculate statistics for each bucket"""
        stats = []
        all_boundaries = [self.MIN_FICO] + boundaries + [self.MAX_FICO + 1]
        n_buckets = len(boundaries) + 1
        
        for i in range(n_buckets):
            lower = all_boundaries[i]
            upper = all_boundaries[i + 1] - 1 if i < n_buckets - 1 else self.MAX_FICO
            
            mask = (self.fico_values >= lower) & (self.fico_values <= upper)
            fico_bucket = self.fico_values[mask]
            default_bucket = self.default_labels[mask]
            
            n_total = len(fico_bucket)
            n_defaults = np.sum(default_bucket) if n_total > 0 else 0
            pd_rate = n_defaults / n_total if n_total > 0 else 0
            
            stats.append({
                'bucket_lower': lower,
                'bucket_upper': upper,
                'n_records': n_total,
                'n_defaults': int(n_defaults),
                'default_rate': pd_rate,
                'mean_fico': np.mean(fico_bucket) if n_total > 0 else None,
                'std_fico': np.std(fico_bucket) if n_total > 0 else None,
                'min_fico': np.min(fico_bucket) if n_total > 0 else None,
                'max_fico': np.max(fico_bucket) if n_total > 0 else None
            })
        
        return pd.DataFrame(stats)
    
    def plot_buckets(self, result: QuantizationResult = None):
        """Visualize the bucket distribution and default rates"""
        if result is None:
            result = self.result
        if result is None:
            raise ValueError("No quantization result available")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Histogram of FICO scores colored by bucket
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, result.n_buckets))
        
        all_boundaries = [self.MIN_FICO] + result.bucket_boundaries + [self.MAX_FICO + 1]
        
        for i in range(result.n_buckets):
            lower = all_boundaries[i]
            upper = all_boundaries[i + 1]
            mask = (self.fico_values >= lower) & (self.fico_values < upper)
            if i == result.n_buckets - 1:
                mask = (self.fico_values >= lower) & (self.fico_values <= self.MAX_FICO)
            
            axes[0].hist(self.fico_values[mask], bins=30, 
                        label=f"{result.bucket_labels[i]} (PD: {result.bucket_stats.iloc[i]['default_rate']:.2%})",
                        alpha=0.7, color=colors[i], edgecolor='black')
        
        axes[0].set_xlabel('FICO Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('FICO Score Distribution by Rating Bucket')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Default rate by bucket
        x_pos = np.arange(result.n_buckets)
        bars = axes[1].bar(x_pos, result.bucket_stats['default_rate'].values, 
                          color=colors, edgecolor='black', alpha=0.8)
        
        # Add value labels on bars
        for bar, rate in zip(bars, result.bucket_stats['default_rate'].values):
            height = bar.get_height()
            axes[1].annotate(f'{rate:.2%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        axes[1].set_xlabel('Rating Bucket (R1 = Best Credit)')
        axes[1].set_ylabel('Probability of Default')
        axes[1].set_title('Default Rate by Rating Bucket')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(result.bucket_labels)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_rating_map(self, result: QuantizationResult = None) -> Dict[str, Dict]:
        """
        Generate a comprehensive rating map for production use.
        
        Returns:
        --------
        Dictionary mapping rating labels to bucket properties
        """
        if result is None:
            result = self.result
        if result is None:
            raise ValueError("No quantization result available")
        
        rating_map = {}
        for idx, row in result.bucket_stats.iterrows():
            rating = result.bucket_labels[idx]
            rating_map[rating] = {
                'fico_range': (int(row['bucket_lower']), int(row['bucket_upper'])),
                'n_records': int(row['n_records']),
                'n_defaults': int(row['n_defaults']),
                'probability_of_default': float(row['default_rate']),
                'mean_fico': float(row['mean_fico']) if row['mean_fico'] is not None else None,
                'description': f"Rating {rating}: FICO {int(row['bucket_lower'])}-{int(row['bucket_upper'])}"
            }
        
        return rating_map


class QuantizationTester:
    """Comprehensive testing suite for FICO quantization"""
    
    def __init__(self, quantizer: FICOQuantizer):
        self.quantizer = quantizer
        self.results = {}
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete test suite"""
        print("=" * 70)
        print("FICO QUANTIZATION TEST SUITE")
        print("=" * 70)
        
        tests = [
            ('Data Loading Test', self.test_data_loading),
            ('Boundary Validity Test', self.test_boundary_validity),
            ('Bucket Coverage Test', self.test_bucket_coverage),
            ('Rating Mapping Test', self.test_rating_mapping),
            ('MSE Optimization Test', self.test_mse_optimization),
            ('LogLikelihood Optimization Test', self.test_loglik_optimization),
            ('DP vs Greedy Comparison', self.test_dp_vs_greedy),
            ('Edge Cases Test', self.test_edge_cases),
            ('Reproducibility Test', self.test_reproducibility),
            ('Performance Test', self.test_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n[{test_name}]")
            try:
                result = test_func()
                self.results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"  Result: {status}")
            except Exception as e:
                self.results[test_name] = False
                print(f"  Result: ❌ FAIL - {e}")
        
        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        passed = sum(self.results.values())
        total = len(self.results)
        print(f"Tests Passed: {passed}/{total} ({100*passed/total:.1f}%)")
        
        for name, result in self.results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {name}")
        
        return self.results
    
    def test_data_loading(self) -> bool:
        """Test that data loads correctly"""
        assert self.quantizer.fico_values is not None, "FICO values not loaded"
        assert self.quantizer.default_labels is not None, "Default labels not loaded"
        assert len(self.quantizer.fico_values) == len(self.quantizer.default_labels), "Data length mismatch"
        assert np.all((self.quantizer.fico_values >= 300) & (self.quantizer.fico_values <= 850)), "FICO out of range"
        assert np.all((self.quantizer.default_labels == 0) | (self.quantizer.default_labels == 1)), "Invalid default labels"
        print(f"  Loaded {len(self.quantizer.fico_values)} records")
        print(f"  FICO range: [{self.quantizer.fico_values.min()}, {self.quantizer.fico_values.max()}]")
        print(f"  Default rate: {np.mean(self.quantizer.default_labels):.2%}")
        return True
    
    def test_boundary_validity(self) -> bool:
        """Test that boundaries are valid"""
        result = self.quantizer.quantize(n_buckets=5, method='mse')
        
        # Check boundaries are sorted
        assert result.bucket_boundaries == sorted(result.bucket_boundaries), "Boundaries not sorted"
        
        # Check boundaries are within range
        for b in result.bucket_boundaries:
            assert 300 <= b <= 850, f"Boundary {b} out of range"
        
        # Check correct number of boundaries
        assert len(result.bucket_boundaries) == result.n_buckets - 1, "Wrong number of boundaries"
        
        # Check no duplicate boundaries
        assert len(result.bucket_boundaries) == len(set(result.bucket_boundaries)), "Duplicate boundaries"
        
        print(f"  Boundaries: {result.bucket_boundaries}")
        return True
    
    def test_bucket_coverage(self) -> bool:
        """Test that all FICO scores are assigned to a bucket"""
        result = self.quantizer.quantize(n_buckets=5)
        
        # Every FICO score should map to a rating
        for fico in self.quantizer.fico_values[:100]:  # Sample for speed
            rating = result.get_rating(fico)
            assert rating is not None, f"FICO {fico} not mapped to any rating"
            assert rating in result.bucket_labels, f"Invalid rating: {rating}"
        
        # Check bucket stats sum to total
        total_in_buckets = result.bucket_stats['n_records'].sum()
        assert total_in_buckets == len(self.quantizer.fico_values), "Bucket coverage incomplete"
        
        print(f"  All {len(self.quantizer.fico_values)} records covered by buckets")
        return True
    
    def test_rating_mapping(self) -> bool:
        """Test that rating mapping is correct (lower = better)"""
        result = self.quantizer.quantize(n_buckets=5)
        
        # R1 should have highest mean FICO, R5 should have lowest
        mean_ficos = result.bucket_stats['mean_fico'].values
        
        # Since we reversed labels, R1 (index 0 after reversal) should have highest FICO
        # Check monotonic decrease
        for i in range(len(mean_ficos) - 1):
            assert mean_ficos[i] >= mean_ficos[i + 1], f"Non-monotonic FICO means: {mean_ficos}"
        
        # Test specific mappings
        high_fico = 800
        low_fico = 350
        
        high_rating = result.get_rating(high_fico)
        low_rating = result.get_rating(low_fico)
        
        high_num = int(high_rating.replace('R', ''))
        low_num = int(low_rating.replace('R', ''))
        
        assert high_num < low_num, f"Rating inversion: high FICO {high_fico}->{high_rating}, low FICO {low_fico}->{low_rating}"
        
        print(f"  High FICO ({high_fico}) -> {high_rating} (better)")
        print(f"  Low FICO ({low_fico}) -> {low_rating} (worse)")
        return True
    
    def test_mse_optimization(self) -> bool:
        """Test MSE optimization produces reasonable results"""
        result_mse = self.quantizer.quantize(n_buckets=5, method='mse')
        result_ll = self.quantizer.quantize(n_buckets=5, method='log_likelihood')
        
        # MSE should be finite and positive
        assert result_mse.objective_value > 0, "MSE objective should be positive"
        assert np.isfinite(result_mse.objective_value), "MSE objective should be finite"
        
        # Each bucket should have reasonable variance
        for idx, row in result_mse.bucket_stats.iterrows():
            if row['n_records'] > 1:
                assert row['std_fico'] is not None, f"Missing std for bucket {idx}"
                assert row['std_fico'] >= 0, f"Negative std for bucket {idx}"
        
        print(f"  MSE Objective: {result_mse.objective_value:.2f}")
        print(f"  Avg bucket std: {result_mse.bucket_stats['std_fico'].mean():.2f}")
        return True
    
    def test_loglik_optimization(self) -> bool:
        """Test log-likelihood optimization"""
        result = self.quantizer.quantize(n_buckets=5, method='log_likelihood')
        
        # Log-likelihood should be finite (negative for Bernoulli)
        assert np.isfinite(result.objective_value), "Log-likelihood should be finite"
        
        # Default rates should be between 0 and 1
        assert np.all((result.bucket_stats['default_rate'] >= 0) & 
                     (result.bucket_stats['default_rate'] <= 1)), "Invalid default rates"
        
        print(f"  Log-Likelihood Objective: {result.objective_value:.2f}")
        print(f"  Default rates: {result.bucket_stats['default_rate'].values}")
        return True
    
    def test_dp_vs_greedy(self) -> bool:
        """Compare DP and greedy optimization"""
        # Use small subset for fair comparison
        n_buckets = 4
        
        result_dp = self.quantizer.quantize(n_buckets=n_buckets, method='mse', use_dp=True)
        result_greedy = self.quantizer.quantize(n_buckets=n_buckets, method='mse', use_dp=False)
        
        # Both should produce valid results
        assert len(result_dp.bucket_boundaries) == n_buckets - 1
        assert len(result_greedy.bucket_boundaries) == n_buckets - 1
        
        # DP should be at least as good as greedy for MSE (minimization)
        # Allow small tolerance for numerical differences
        assert result_dp.objective_value <= result_greedy.objective_value * 1.01, \
            f"DP MSE ({result_dp.objective_value:.2f}) worse than greedy ({result_greedy.objective_value:.2f})"
        
        print(f"  DP MSE: {result_dp.objective_value:.2f}")
        print(f"  Greedy MSE: {result_greedy.objective_value:.2f}")
        return True
    
    def test_edge_cases(self) -> bool:
        """Test edge cases"""
        # Test minimum buckets
        result_2 = self.quantizer.quantize(n_buckets=2)
        assert len(result_2.bucket_boundaries) == 1
        
        # Test maximum reasonable buckets
        result_10 = self.quantizer.quantize(n_buckets=10)
        assert len(result_10.bucket_boundaries) == 9
        
        # Test boundary FICO values
        result = self.quantizer.quantize(n_buckets=5)
        
        # Minimum FICO should be in a bucket
        min_rating = result.get_rating(300)
        assert min_rating is not None, "FICO 300 not mapped"
        
        # Maximum FICO should be in a bucket
        max_rating = result.get_rating(850)
        assert max_rating is not None, "FICO 850 not mapped"
        
        print(f"  FICO 300 -> {min_rating}, FICO 850 -> {max_rating}")
        return True
    
    def test_reproducibility(self) -> bool:
        """Test that results are reproducible"""
        result1 = self.quantizer.quantize(n_buckets=5, method='mse')
        result2 = self.quantizer.quantize(n_buckets=5, method='mse')
        
        # Boundaries should be identical
        assert result1.bucket_boundaries == result2.bucket_boundaries, "Non-reproducible boundaries"
        
        # Objective values should match
        assert abs(result1.objective_value - result2.objective_value) < 1e-6, "Non-reproducible objective"
        
        print("  Results are reproducible")
        return True
    
    def test_performance(self) -> bool:
        """Basic performance test"""
        import time
        
        n_buckets = 5
        
        start = time.time()
        result = self.quantizer.quantize(n_buckets=n_buckets, method='mse', use_dp=True)
        dp_time = time.time() - start
        
        start = time.time()
        result = self.quantizer.quantize(n_buckets=n_buckets, method='mse', use_dp=False)
        greedy_time = time.time() - start
        
        # Should complete in reasonable time (< 30 seconds for DP, < 5 for greedy)
        assert dp_time < 30, f"DP too slow: {dp_time:.2f}s"
        assert greedy_time < 5, f"Greedy too slow: {greedy_time:.2f}s"
        
        print(f"  DP time: {dp_time:.2f}s, Greedy time: {greedy_time:.2f}s")
        return True


def load_loan_data(filepath: str = 'Task 3 and 4_Loan_Data (1).csv') -> pd.DataFrame:
    """Load and preprocess loan data"""
    df = pd.read_csv(filepath)
    
    # Handle potential parsing issues
    if 'fico_score' not in df.columns:
        # Try to find FICO column
        fico_cols = [c for c in df.columns if 'fico' in c.lower()]
        if fico_cols:
            df = df.rename(columns={fico_cols[0]: 'fico_score'})
    
    if 'default' not in df.columns:
        # Try to find default column
        default_cols = [c for c in df.columns if 'default' in c.lower()]
        if default_cols:
            df = df.rename(columns={default_cols[0]: 'default'})
    
    # Ensure correct types
    df['fico_score'] = pd.to_numeric(df['fico_score'], errors='coerce')
    df['default'] = pd.to_numeric(df['default'], errors='coerce').fillna(0).astype(int)
    
    # Remove invalid records
    df = df.dropna(subset=['fico_score'])
    df = df[(df['fico_score'] >= 300) & (df['fico_score'] <= 850)]
    
    return df


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("FICO SCORE QUANTIZATION & RATING MAP GENERATOR")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading loan data...")
    df = load_loan_data()
    print(f"  Loaded {len(df)} records with FICO scores")
    
    # Initialize quantizer
    print("\n[2/4] Initializing FICO quantizer...")
    quantizer = FICOQuantizer(data=df)
    
    # Run tests
    print("\n[3/4] Running test suite...")
    tester = QuantizationTester(quantizer)
    test_results = tester.run_all_tests()
    
    # Generate optimal quantization
    print("\n[4/4] Generating optimal rating map...")
    
    # Try both methods and compare
    print("\n" + "-" * 70)
    print("COMPARING OPTIMIZATION METHODS (5 buckets)")
    print("-" * 70)
    
    result_mse = quantizer.quantize(n_buckets=5, method='mse')
    result_ll = quantizer.quantize(n_buckets=5, method='log_likelihood')
    
    print(f"\nMSE Optimization:")
    print(f"  Objective: {result_mse.objective_value:.2f}")
    print(f"  Boundaries: {result_mse.bucket_boundaries}")
    print(f"\nLog-Likelihood Optimization:")
    print(f"  Objective: {result_ll.objective_value:.2f}")
    print(f"  Boundaries: {result_ll.bucket_boundaries}")
    
    # Display bucket statistics
    print("\n" + "-" * 70)
    print("RATING BUCKET STATISTICS (Log-Likelihood Method)")
    print("-" * 70)
    print(f"{'Rating':<8} {'FICO Range':<15} {'Records':<10} {'Defaults':<10} {'PD Rate':<10}")
    print("-" * 70)
    
    for idx, row in result_ll.bucket_stats.iterrows():
        rating = result_ll.bucket_labels[idx]
        fico_range = f"{int(row['bucket_lower'])}-{int(row['bucket_upper'])}"
        print(f"{rating:<8} {fico_range:<15} {int(row['n_records']):<10} "
              f"{int(row['n_defaults']):<10} {row['default_rate']:.2%}")
    
    # Generate production rating map
    rating_map = quantizer.generate_rating_map(result_ll)
    
    print("\n" + "=" * 70)
    print("PRODUCTION RATING MAP")
    print("=" * 70)
    for rating, props in rating_map.items():
        print(f"\n{rating}:")
        print(f"  FICO Range: {props['fico_range'][0]} - {props['fico_range'][1]}")
        print(f"  Probability of Default: {props['probability_of_default']:.2%}")
        print(f"  Sample Size: {props['n_records']} records")
    
    # Visualization (optional)
    print("\n" + "-" * 70)
    print("Generating visualization...")
    print("-" * 70)
    try:
        quantizer.plot_buckets(result_ll)
        print("  ✅ Visualization complete")
    except Exception as e:
        print(f"  ⚠ Visualization skipped: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)
    print(f"✅ All tests passed: {all(test_results.values())}")
    print(f"✅ Rating map generated with {len(rating_map)} buckets")
    print(f"✅ Ready for production use")
    print("=" * 70)
    
    return {
        'quantizer': quantizer,
        'result': result_ll,
        'rating_map': rating_map,
        'test_results': test_results
    }


# ============================================================================
# PRODUCTION HELPER FUNCTIONS
# ============================================================================

def create_fico_rating(fico_score: float, boundaries: List[float], 
                      labels: List[str]) -> str:
    """
    Production helper: Map FICO score to rating.
    
    Parameters:
    -----------
    fico_score : float
        Borrower's FICO score
    boundaries : List[float]
        Bucket boundary values from quantization
    labels : List[str]
        Rating labels (R1, R2, ...) where R1 = best
    
    Returns:
    --------
    str : Rating label
    """
    all_bounds = [300] + boundaries + [851]
    
    for i, (lower, upper) in enumerate(zip(all_bounds[:-1], all_bounds[1:])):
        if lower <= fico_score < upper or (i == len(labels) - 1 and fico_score == upper - 1):
            return labels[i]
    
    # Fallback for edge cases
    if fico_score <= boundaries[0]:
        return labels[-1]  # Worst rating for low FICO
    elif fico_score >= boundaries[-1]:
        return labels[0]  # Best rating for high FICO
    return labels[len(labels) // 2]  # Middle rating as default


def calculate_bucket_pd(fico_score: float, rating_map: Dict) -> float:
    """
    Production helper: Get probability of default for a FICO score.
    
    Parameters:
    -----------
    fico_score : float
        Borrower's FICO score
    rating_map : Dict
        Rating map from generate_rating_map()
    
    Returns:
    --------
    float : Probability of default
    """
    for rating, props in rating_map.items():
        lower, upper = props['fico_range']
        if lower <= fico_score <= upper:
            return props['probability_of_default']
    return None


if __name__ == "__main__":
    results = main()
    
    # Example usage for production
    print("\n" + "=" * 70)
    print("PRODUCTION USAGE EXAMPLES")
    print("=" * 70)
    
    rating_map = results['rating_map']
    result = results['result']
    
    # Example 1: Rate a new borrower
    new_fico = 720
    rating = create_fico_rating(new_fico, result.bucket_boundaries, result.bucket_labels)
    pd = calculate_bucket_pd(new_fico, rating_map)
    
    print(f"\nNew borrower with FICO {new_fico}:")
    print(f"  Rating: {rating}")
    print(f"  Estimated PD: {pd:.2%}")
    
    # Example 2: Portfolio analysis
    sample_ficos = [350, 500, 650, 750, 820]
    print(f"\nPortfolio sample analysis:")
    print(f"{'FICO':<6} {'Rating':<8} {'PD':<10}")
    print("-" * 30)
    for fico in sample_ficos:
        r = create_fico_rating(fico, result.bucket_boundaries, result.bucket_labels)
        p = calculate_bucket_pd(fico, rating_map)
        print(f"{fico:<6} {r:<8} {p:.2%}")
    
    print("\n" + "=" * 70)