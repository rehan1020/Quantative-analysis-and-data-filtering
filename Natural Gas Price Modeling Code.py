import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Check if statsmodels is available
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

class NaturalGasPriceModel:
    def __init__(self, file_path='Nat_Gas.csv'):
        """
        Initialize the model by loading and preprocessing data.
        """
        self.df = None
        self.forecast_df = None
        self.model = None
        self.load_data(file_path)
        if STATSMODELS_AVAILABLE:
            self.train_model()
        else:
            print("Forecasting disabled due to missing statsmodels library.")
        
    def load_data(self, file_path):
        """
        Load CSV, parse dates, and handle scientific notation in prices.
        """
        try:
            # Load data
            self.df = pd.read_csv(file_path)
            
            # Parse Dates - handle MM/DD/YY format
            self.df['Dates'] = pd.to_datetime(self.df['Dates'], format='%m/%d/%y')
            
            # Parse Prices - pandas handles scientific notation automatically
            self.df['Prices'] = pd.to_numeric(self.df['Prices'], errors='coerce')
            
            # Remove any rows with NaN prices
            self.df = self.df.dropna(subset=['Prices'])
            
            # Sort by date
            self.df = self.df.sort_values('Dates').reset_index(drop=True)
            
            # Set date as index for time series operations
            self.df.set_index('Dates', inplace=True)
            
            # Ensure monthly frequency
            self.df = self.df.asfreq('M')
            
            # Interpolate any missing months
            self.df['Prices'] = self.df['Prices'].interpolate(method='linear')
            
            # Mark as historical
            self.df['Type'] = 'Historical'
            
            print(f"✓ Data loaded successfully.")
            print(f"  Range: {self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}")
            print(f"  Total Observations: {len(self.df)}")
            print(f"  Price Range: {self.df['Prices'].min():.2f} to {self.df['Prices'].max():.2f}")
            
        except FileNotFoundError:
            print(f"✗ Error: File '{file_path}' not found. Please ensure the file is in the working directory.")
            raise
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise

    def train_model(self):
        """
        Train Holt-Winters Exponential Smoothing model for forecasting.
        Captures Trend and Seasonality.
        """
        try:
            # Ensure no NaNs in training data
            train_data = self.df['Prices'].dropna()
            
            if len(train_data) < 24:
                print("⚠ Warning: Less than 2 years of data. Seasonal modeling may be less accurate.")
            
            # Fit Holt-Winters Model
            self.model = ExponentialSmoothing(
                train_data, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            ).fit()
            
            # Forecast 12 months into the future
            last_date = self.df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.offsets.MonthEnd(1), 
                periods=12, 
                freq='M'
            )
            
            forecast_values = self.model.forecast(steps=12)
            
            self.forecast_df = pd.DataFrame({
                'Prices': forecast_values.values,
                'Type': 'Forecast'
            }, index=future_dates)
            
            print(f"✓ Model training complete.")
            print(f"  Forecast Range: {self.forecast_df.index.min().strftime('%Y-%m-%d')} to {self.forecast_df.index.max().strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"✗ Error training model: {e}")
            print("  Falling back to simple linear trend forecast.")
            self._simple_forecast()

    def _simple_forecast(self):
        """
        Fallback simple linear trend forecast if Holt-Winters fails.
        """
        # Calculate average monthly change
        prices = self.df['Prices'].dropna()
        n_months = len(prices)
        if n_months > 1:
            avg_change = (prices.iloc[-1] - prices.iloc[0]) / (n_months - 1)
        else:
            avg_change = 0
        
        last_date = self.df.index[-1]
        last_price = prices.iloc[-1]
        
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1), 
            periods=12, 
            freq='M'
        )
        
        forecast_values = [last_price + avg_change * (i + 1) for i in range(12)]
        
        self.forecast_df = pd.DataFrame({
            'Prices': forecast_values,
            'Type': 'Forecast'
        }, index=future_dates)
        
        print("✓ Simple linear forecast generated.")

    def get_price_estimate(self, input_date):
        """
        Takes a date string or datetime object and returns an estimated price.
        Interpolates for past, extrapolates for future.
        """
        # Parse input date
        if isinstance(input_date, str):
            target_date = self._parse_date(input_date)
            if target_date is None:
                return "Error: Invalid date format. Please use YYYY-MM-DD, MM/DD/YYYY, or MM/DD/YY."
        else:
            target_date = pd.to_datetime(input_date)
        
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        
        # Check if date is before available data
        if target_date < start_date:
            return f"Error: Date {target_date.strftime('%Y-%m-%d')} is before available data range ({start_date.strftime('%Y-%m-%d')})."
        
        # Historical/Interpolation range
        elif target_date <= end_date:
            estimate = self._interpolate_price(target_date)
            return f"Date: {target_date.strftime('%Y-%m-%d')} | Estimated Price: {estimate:.4f} | Type: Interpolated"
        
        # Forecast range
        elif self.forecast_df is not None and target_date <= self.forecast_df.index.max():
            # Find closest forecast date (month-end)
            forecast_date = target_date + pd.offsets.MonthEnd(0)
            if forecast_date in self.forecast_df.index:
                estimate = self.forecast_df.loc[forecast_date, 'Prices']
                return f"Date: {target_date.strftime('%Y-%m-%d')} | Estimated Price: {estimate:.4f} | Type: Forecast"
            else:
                # Interpolate between forecast months if needed
                estimate = self._interpolate_forecast(target_date)
                return f"Date: {target_date.strftime('%Y-%m-%d')} | Estimated Price: {estimate:.4f} | Type: Forecast (Interpolated)"
        
        # Beyond forecast range
        else:
            forecast_end = self.forecast_df.index.max() if self.forecast_df is not None else end_date
            return f"Warning: Date {target_date.strftime('%Y-%m-%d')} is beyond the 1-year forecast horizon (ends {forecast_end.strftime('%Y-%m-%d')}). Model uncertainty increases significantly."

    def _parse_date(self, date_str):
        """
        Parse date string with multiple format attempts.
        """
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m/%d/%y',
            '%d/%m/%Y',
            '%Y/%m/%d'
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Try pandas default parser as last resort
        try:
            return pd.to_datetime(date_str)
        except:
            return None

    def _interpolate_price(self, target_date):
        """
        Interpolate price for historical dates.
        """
        # Create a temporary series with the target date
        temp_series = self.df['Prices'].copy()
        temp_series.loc[target_date] = np.nan
        temp_series = temp_series.sort_index()
        
        # Interpolate using time-based method
        temp_series = temp_series.interpolate(method='time')
        
        return temp_series.loc[target_date]

    def _interpolate_forecast(self, target_date):
        """
        Interpolate price within forecast range.
        """
        # Combine historical and forecast data
        combined = pd.concat([self.df['Prices'], self.forecast_df['Prices']])
        combined = combined.sort_index()
        
        # Add target date and interpolate
        combined.loc[target_date] = np.nan
        combined = combined.sort_index()
        combined = combined.interpolate(method='time')
        
        return combined.loc[target_date]

    def visualize_data(self, save_path=None):
        """
        Generate plots for trend analysis and seasonality.
        """
        fig, axs = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Historical vs Forecast
        axs[0].plot(self.df.index, self.df['Prices'], 
                   label='Historical Snapshot', marker='o', 
                   color='#2E86AB', linewidth=2, markersize=6)
        
        if self.forecast_df is not None:
            axs[0].plot(self.forecast_df.index, self.forecast_df['Prices'], 
                       label='12-Month Forecast', marker='x', 
                       color='#A23B72', linestyle='--', linewidth=2, markersize=8)
        
        axs[0].set_title('Natural Gas Price History and 1-Year Forecast', fontsize=14, fontweight='bold')
        axs[0].set_xlabel('Date', fontsize=12)
        axs[0].set_ylabel('Price Units', fontsize=12)
        axs[0].legend(loc='best', fontsize=10)
        axs[0].grid(True, alpha=0.3, linestyle='--')
        axs[0].tick_params(axis='both', labelsize=10)
        
        # Plot 2: Seasonality Check (Boxplot by Month)
        self.df['Month_Num'] = self.df.index.month
        self.df['Month_Name'] = self.df.index.month_name()
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        # Prepare data for boxplot
        plot_data = []
        for m in month_order:
            month_data = self.df[self.df['Month_Name'] == m]['Prices'].dropna()
            if len(month_data) > 0:
                plot_data.append(month_data.values)
            else:
                plot_data.append([np.nan])
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, 12))
        bp = axs[1].boxplot(plot_data, labels=month_order, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axs[1].set_title('Seasonal Price Distribution (2020-2024)', fontsize=14, fontweight='bold')
        axs[1].set_xlabel('Month', fontsize=12)
        axs[1].set_ylabel('Price Units', fontsize=12)
        axs[1].tick_params(axis='x', rotation=45, labelsize=9)
        axs[1].tick_params(axis='y', labelsize=10)
        axs[1].grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()

    def get_summary_statistics(self):
        """
        Return summary statistics of the historical data.
        """
        stats = {
            'Mean Price': self.df['Prices'].mean(),
            'Median Price': self.df['Prices'].median(),
            'Min Price': self.df['Prices'].min(),
            'Max Price': self.df['Prices'].max(),
            'Std Deviation': self.df['Prices'].std(),
            'Start Date': self.df.index.min().strftime('%Y-%m-%d'),
            'End Date': self.df.index.max().strftime('%Y-%m-%d'),
            'Total Months': len(self.df)
        }
        
        if self.forecast_df is not None:
            stats['Forecast Mean'] = self.forecast_df['Prices'].mean()
            stats['Forecast Min'] = self.forecast_df['Prices'].min()
            stats['Forecast Max'] = self.forecast_df['Prices'].max()
        
        return stats


# --- Execution Block ---
if __name__ == "__main__":
    print("=" * 70)
    print("NATURAL GAS PRICE MODELING & FORECASTING SYSTEM")
    print("=" * 70)
    print()
    
    # Initialize Model
    try:
        model = NaturalGasPriceModel('Nat_Gas.csv')
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        exit(1)
    
    print()
    
    # Display Summary Statistics
    print("-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    stats = model.get_summary_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Visualize Patterns
    print("-" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("-" * 70)
    model.visualize_data()
    print()
    
    # Example Queries
    print("-" * 70)
    print("PRICE ESTIMATE EXAMPLES")
    print("-" * 70)
    test_dates = [
        '03/15/2022',   # Historical - interpolation
        '12/31/2023',   # Historical - exact match
        '05/30/2025',   # Future - forecast
        '2024-06-15',   # Alternative date format
        '01/15/2020'    # Before data range - error case
    ]
    
    for date in test_dates:
        result = model.get_price_estimate(date)
        print(f"  {result}")
    
    print()
    print("=" * 70)
    print("MODEL READY FOR USE")
    print("=" * 70)
    print()
    print("To query a specific date, use:")
    print("  model.get_price_estimate('YYYY-MM-DD') or model.get_price_estimate('MM/DD/YYYY')")
    print()
    print("DISCLAIMER: This model is for indicative purposes only.")
    print("Natural gas markets are volatile. Do not use for actual trading decisions")
    print("without additional fundamental analysis and risk management.")
    print("=" * 70)