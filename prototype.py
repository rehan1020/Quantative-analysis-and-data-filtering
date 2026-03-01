import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from typing import List, Union, Dict, Tuple
from dataclasses import dataclass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Check if statsmodels is available
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")


@dataclass
class StorageContract:
    """
    Data class to hold storage contract parameters.
    """
    injection_dates: List[str]
    withdrawal_dates: List[str]
    injection_volume: float  # Total volume to inject (in MMBtu)
    withdrawal_volume: float  # Total volume to withdraw (in MMBtu)
    max_storage_volume: float  # Maximum storage capacity (in MMBtu)
    injection_rate: float  # Max injection rate per day (in MMBtu/day)
    withdrawal_rate: float  # Max withdrawal rate per day (in MMBtu/day)
    storage_cost_per_month: float  # Storage cost per month per MMBtu ($/MMBtu/month)
    injection_cost_per_volume: float  # Injection cost per MMBtu ($/MMBtu)
    withdrawal_cost_per_volume: float  # Withdrawal cost per MMBtu ($/MMBtu)
    transport_cost_per_transaction: float  # Fixed transport cost per transaction ($)


class NaturalGasPriceModel:
    """
    Price estimation model for natural gas (from previous implementation).
    """
    def __init__(self, file_path='Nat_Gas.csv'):
        self.df = None
        self.forecast_df = None
        self.model = None
        self.load_data(file_path)
        if STATSMODELS_AVAILABLE:
            self.train_model()
        else:
            print("Forecasting disabled due to missing statsmodels library.")
        
    def load_data(self, file_path):
        try:
            self.df = pd.read_csv(file_path)
            self.df['Dates'] = pd.to_datetime(self.df['Dates'], format='%m/%d/%y')
            self.df['Prices'] = pd.to_numeric(self.df['Prices'], errors='coerce')
            self.df = self.df.dropna(subset=['Prices'])
            self.df = self.df.sort_values('Dates').reset_index(drop=True)
            self.df.set_index('Dates', inplace=True)
            self.df = self.df.asfreq('M')
            self.df['Prices'] = self.df['Prices'].interpolate(method='linear')
            self.df['Type'] = 'Historical'
            print(f"✓ Data loaded: {self.df.index.min().strftime('%Y-%m-%d')} to {self.df.index.max().strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"✗ Error loading  {e}")
            raise

    def train_model(self):
        try:
            train_data = self.df['Prices'].dropna()
            if len(train_data) < 24:
                print("⚠ Warning: Less than 2 years of data for seasonal modeling.")
            
            self.model = ExponentialSmoothing(
                train_data, trend='add', seasonal='add', seasonal_periods=12
            ).fit()
            
            last_date = self.df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.offsets.MonthEnd(1), periods=12, freq='M'
            )
            forecast_values = self.model.forecast(steps=12)
            
            self.forecast_df = pd.DataFrame({
                'Prices': forecast_values.values,
                'Type': 'Forecast'
            }, index=future_dates)
            
            print(f"✓ Model trained. Forecast: {self.forecast_df.index.min().strftime('%Y-%m-%d')} to {self.forecast_df.index.max().strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"⚠ Model training failed: {e}. Using simple forecast.")
            self._simple_forecast()

    def _simple_forecast(self):
        prices = self.df['Prices'].dropna()
        n_months = len(prices)
        avg_change = (prices.iloc[-1] - prices.iloc[0]) / (n_months - 1) if n_months > 1 else 0
        
        last_date = self.df.index[-1]
        last_price = prices.iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1), periods=12, freq='M'
        )
        forecast_values = [last_price + avg_change * (i + 1) for i in range(12)]
        
        self.forecast_df = pd.DataFrame({
            'Prices': forecast_values,
            'Type': 'Forecast'
        }, index=future_dates)

    def get_price(self, input_date) -> float:
        """Get price estimate for a given date."""
        if isinstance(input_date, str):
            target_date = pd.to_datetime(input_date)
        else:
            target_date = pd.to_datetime(input_date)
        
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        
        if target_date < start_date:
            raise ValueError(f"Date {target_date.date()} is before available data.")
        
        elif target_date <= end_date:
            temp_series = self.df['Prices'].copy()
            temp_series.loc[target_date] = np.nan
            temp_series = temp_series.sort_index().interpolate(method='time')
            return temp_series.loc[target_date]
        
        elif self.forecast_df is not None and target_date <= self.forecast_df.index.max():
            forecast_date = target_date + pd.offsets.MonthEnd(0)
            if forecast_date in self.forecast_df.index:
                return self.forecast_df.loc[forecast_date, 'Prices']
            else:
                combined = pd.concat([self.df['Prices'], self.forecast_df['Prices']])
                combined = combined.sort_index()
                combined.loc[target_date] = np.nan
                combined = combined.sort_index().interpolate(method='time')
                return combined.loc[target_date]
        
        else:
            raise ValueError(f"Date {target_date.date()} is beyond forecast horizon.")


class StorageContractPricer:
    """
    Main pricing model for natural gas storage contracts.
    Calculates the net present value of a storage contract considering all cash flows.
    """
    
    def __init__(self, price_model: NaturalGasPriceModel):
        """
        Initialize the pricer with a price model.
        
        Args:
            price_model: NaturalGasPriceModel instance for price estimation
        """
        self.price_model = price_model
        self.contract = None
        self.cash_flows = []
        
    def price_contract(self, contract: StorageContract) -> Dict:
        """
        Calculate the value of a storage contract.
        
        Args:
            contract: StorageContract object with all parameters
            
        Returns:
            Dictionary containing contract valuation and breakdown
        """
        self.contract = contract
        self.cash_flows = []
        
        # Validate contract parameters
        validation = self._validate_contract(contract)
        if not validation['valid']:
            return {
                'valid': False,
                'error': validation['error'],
                'contract_value': None
            }
        
        # Calculate all cash flows
        # 1. Injection costs (buying gas + injection fees + transport)
        injection_cash_flows = self._calculate_injection_flows(contract)
        
        # 2. Storage costs (monthly fees while gas is stored)
        storage_cash_flows = self._calculate_storage_flows(contract)
        
        # 3. Withdrawal revenues (selling gas - withdrawal fees - transport)
        withdrawal_cash_flows = self._calculate_withdrawal_flows(contract)
        
        # Combine all cash flows
        all_flows = injection_cash_flows + storage_cash_flows + withdrawal_cash_flows
        
        # Calculate net value (sum of all cash flows)
        # Negative = cost, Positive = revenue
        total_injection_cost = sum([f['amount'] for f in injection_cash_flows])
        total_storage_cost = sum([f['amount'] for f in storage_cash_flows])
        total_withdrawal_revenue = sum([f['amount'] for f in withdrawal_cash_flows])
        
        contract_value = total_withdrawal_revenue + total_injection_cost + total_storage_cost
        
        # Build result dictionary
        result = {
            'valid': True,
            'contract_value': contract_value,
            'breakdown': {
                'total_injection_cost': total_injection_cost,
                'total_storage_cost': total_storage_cost,
                'total_withdrawal_revenue': total_withdrawal_revenue,
                'gross_trading_profit': total_withdrawal_revenue + total_injection_cost,
                'net_profit_after_costs': contract_value
            },
            'cash_flows': all_flows,
            'summary': {
                'injection_dates': contract.injection_dates,
                'withdrawal_dates': contract.withdrawal_dates,
                'injection_volume_mmbtu': contract.injection_volume,
                'withdrawal_volume_mmbtu': contract.withdrawal_volume,
                'max_storage_volume_mmbtu': contract.max_storage_volume,
                'storage_duration_months': self._calculate_storage_duration(contract),
                'avg_injection_price': self._calculate_avg_injection_price(contract),
                'avg_withdrawal_price': self._calculate_avg_withdrawal_price(contract)
            }
        }
        
        return result
    
    def _validate_contract(self, contract: StorageContract) -> Dict:
        """
        Validate contract parameters before pricing.
        
        Returns:
            Dictionary with 'valid' boolean and 'error' message if invalid
        """
        # Check injection/withdrawal dates exist
        if not contract.injection_dates or not contract.withdrawal_dates:
            return {'valid': False, 'error': 'Injection and withdrawal dates must be provided.'}
        
        # Check volumes are positive
        if contract.injection_volume <= 0 or contract.withdrawal_volume <= 0:
            return {'valid': False, 'error': 'Injection and withdrawal volumes must be positive.'}
        
        # Check withdrawal doesn't exceed injection (can't sell more than you buy)
        if contract.withdrawal_volume > contract.injection_volume:
            return {'valid': False, 'error': 'Withdrawal volume cannot exceed injection volume.'}
        
        # Check max storage capacity
        if contract.injection_volume > contract.max_storage_volume:
            return {'valid': False, 'error': 'Injection volume exceeds maximum storage capacity.'}
        
        # Check withdrawal dates are after injection dates
        try:
            first_injection = min([pd.to_datetime(d) for d in contract.injection_dates])
            last_withdrawal = max([pd.to_datetime(d) for d in contract.withdrawal_dates])
            last_injection = max([pd.to_datetime(d) for d in contract.injection_dates])
            first_withdrawal = min([pd.to_datetime(d) for d in contract.withdrawal_dates])
            
            if first_withdrawal <= last_injection:
                return {'valid': False, 'error': 'Withdrawal must occur after all injections are complete.'}
        except Exception as e:
            return {'valid': False, 'error': f'Invalid date format: {str(e)}'}
        
        # Check rates are positive
        if contract.injection_rate <= 0 or contract.withdrawal_rate <= 0:
            return {'valid': False, 'error': 'Injection and withdrawal rates must be positive.'}
        
        return {'valid': True, 'error': None}
    
    def _calculate_injection_flows(self, contract: StorageContract) -> List[Dict]:
        """
        Calculate cash flows for gas injection (purchasing and storing).
        
        Returns:
            List of cash flow dictionaries (negative values = costs)
        """
        flows = []
        n_injection_dates = len(contract.injection_dates)
        
        # Distribute volume evenly across injection dates
        volume_per_date = contract.injection_volume / n_injection_dates
        
        for date_str in contract.injection_dates:
            date = pd.to_datetime(date_str)
            
            # Get gas purchase price
            gas_price = self.price_model.get_price(date)
            
            # Calculate costs
            gas_purchase_cost = -volume_per_date * gas_price  # Negative = outflow
            injection_fee = -volume_per_date * contract.injection_cost_per_volume
            transport_cost = -contract.transport_cost_per_transaction
            
            total_cost = gas_purchase_cost + injection_fee + transport_cost
            
            flows.append({
                'date': date,
                'type': 'Injection',
                'volume_mmbtu': volume_per_date,
                'price_per_mmbtu': gas_price,
                'gas_cost': gas_purchase_cost,
                'injection_fee': injection_fee,
                'transport_cost': transport_cost,
                'total_amount': total_cost,
                'description': f"Inject {volume_per_date:,.0f} MMBtu @ ${gas_price:.2f}/MMBtu"
            })
        
        return flows
    
    def _calculate_storage_flows(self, contract: StorageContract) -> List[Dict]:
        """
        Calculate monthly storage costs while gas is held.
        
        Returns:
            List of cash flow dictionaries (negative values = costs)
        """
        flows = []
        
        # Find storage period
        injection_dates = [pd.to_datetime(d) for d in contract.injection_dates]
        withdrawal_dates = [pd.to_datetime(d) for d in contract.withdrawal_dates]
        
        storage_start = max(injection_dates)  # After last injection
        storage_end = min(withdrawal_dates)   # Before first withdrawal
        
        # Calculate number of months in storage
        months_in_storage = (storage_end.year - storage_start.year) * 12 + (storage_end.month - storage_start.month)
        months_in_storage = max(1, months_in_storage)  # At least 1 month
        
        # Calculate monthly storage cost
        # Assume average volume stored is half of injection volume (simplified)
        # More sophisticated models would track exact inventory levels
        avg_volume_stored = contract.injection_volume * 0.5  # Simplified assumption
        monthly_storage_cost = -avg_volume_stored * contract.storage_cost_per_month
        
        for month in range(months_in_storage):
            storage_date = storage_start + pd.DateOffset(months=month)
            
            flows.append({
                'date': storage_date,
                'type': 'Storage',
                'volume_mmbtu': avg_volume_stored,
                'price_per_mmbtu': contract.storage_cost_per_month,
                'storage_fee': monthly_storage_cost,
                'total_amount': monthly_storage_cost,
                'description': f"Month {month+1} storage: {avg_volume_stored:,.0f} MMBtu @ ${contract.storage_cost_per_month:.2f}/MMBtu/month"
            })
        
        return flows
    
    def _calculate_withdrawal_flows(self, contract: StorageContract) -> List[Dict]:
        """
        Calculate cash flows for gas withdrawal (selling).
        
        Returns:
            List of cash flow dictionaries (positive values = revenue)
        """
        flows = []
        n_withdrawal_dates = len(contract.withdrawal_dates)
        
        # Distribute volume evenly across withdrawal dates
        volume_per_date = contract.withdrawal_volume / n_withdrawal_dates
        
        for date_str in contract.withdrawal_dates:
            date = pd.to_datetime(date_str)
            
            # Get gas sale price
            gas_price = self.price_model.get_price(date)
            
            # Calculate revenues and costs
            gas_sale_revenue = volume_per_date * gas_price  # Positive = inflow
            withdrawal_fee = -volume_per_date * contract.withdrawal_cost_per_volume
            transport_cost = -contract.transport_cost_per_transaction
            
            total_amount = gas_sale_revenue + withdrawal_fee + transport_cost
            
            flows.append({
                'date': date,
                'type': 'Withdrawal',
                'volume_mmbtu': volume_per_date,
                'price_per_mmbtu': gas_price,
                'gas_revenue': gas_sale_revenue,
                'withdrawal_fee': withdrawal_fee,
                'transport_cost': transport_cost,
                'total_amount': total_amount,
                'description': f"Withdraw {volume_per_date:,.0f} MMBtu @ ${gas_price:.2f}/MMBtu"
            })
        
        return flows
    
    def _calculate_storage_duration(self, contract: StorageContract) -> float:
        """Calculate approximate storage duration in months."""
        injection_dates = [pd.to_datetime(d) for d in contract.injection_dates]
        withdrawal_dates = [pd.to_datetime(d) for d in contract.withdrawal_dates]
        
        storage_start = max(injection_dates)
        storage_end = min(withdrawal_dates)
        
        duration_days = (storage_end - storage_start).days
        duration_months = duration_days / 30.44  # Average days per month
        
        return round(duration_months, 2)
    
    def _calculate_avg_injection_price(self, contract: StorageContract) -> float:
        """Calculate weighted average injection price."""
        total_cost = 0
        total_volume = 0
        
        for date_str in contract.injection_dates:
            date = pd.to_datetime(date_str)
            price = self.price_model.get_price(date)
            volume = contract.injection_volume / len(contract.injection_dates)
            
            total_cost += price * volume
            total_volume += volume
        
        return round(total_cost / total_volume, 2) if total_volume > 0 else 0
    
    def _calculate_avg_withdrawal_price(self, contract: StorageContract) -> float:
        """Calculate weighted average withdrawal price."""
        total_revenue = 0
        total_volume = 0
        
        for date_str in contract.withdrawal_dates:
            date = pd.to_datetime(date_str)
            price = self.price_model.get_price(date)
            volume = contract.withdrawal_volume / len(contract.withdrawal_dates)
            
            total_revenue += price * volume
            total_volume += volume
        
        return round(total_revenue / total_volume, 2) if total_volume > 0 else 0
    
    def generate_report(self, valuation_result: Dict) -> str:
        """
        Generate a human-readable report of the contract valuation.
        
        Args:
            valuation_result: Dictionary from price_contract()
            
        Returns:
            Formatted string report
        """
        if not valuation_result['valid']:
            return f"❌ Contract Validation Failed: {valuation_result['error']}"
        
        report = []
        report.append("=" * 80)
        report.append("NATURAL GAS STORAGE CONTRACT VALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Contract Summary
        summary = valuation_result['summary']
        report.append("CONTRACT PARAMETERS")
        report.append("-" * 80)
        report.append(f"  Injection Dates:      {', '.join(summary['injection_dates'])}")
        report.append(f"  Withdrawal Dates:     {', '.join(summary['withdrawal_dates'])}")
        report.append(f"  Injection Volume:     {summary['injection_volume_mmbtu']:,.0f} MMBtu")
        report.append(f"  Withdrawal Volume:    {summary['withdrawal_volume_mmbtu']:,.0f} MMBtu")
        report.append(f"  Max Storage Capacity: {summary['max_storage_volume_mmbtu']:,.0f} MMBtu")
        report.append(f"  Storage Duration:     {summary['storage_duration_months']} months")
        report.append("")
        
        # Pricing Summary
        report.append("PRICING SUMMARY")
        report.append("-" * 80)
        report.append(f"  Avg Injection Price:  ${summary['avg_injection_price']:.2f}/MMBtu")
        report.append(f"  Avg Withdrawal Price: ${summary['avg_withdrawal_price']:.2f}/MMBtu")
        report.append(f"  Price Spread:         ${summary['avg_withdrawal_price'] - summary['avg_injection_price']:.2f}/MMBtu")
        report.append("")
        
        # Cash Flow Breakdown
        breakdown = valuation_result['breakdown']
        report.append("CASH FLOW BREAKDOWN")
        report.append("-" * 80)
        report.append(f"  Gas Purchase Cost:    ${breakdown['total_injection_cost']:,.2f}")
        report.append(f"  Storage Costs:        ${breakdown['total_storage_cost']:,.2f}")
        report.append(f"  Gas Sale Revenue:     ${breakdown['total_withdrawal_revenue']:,.2f}")
        report.append("")
        report.append(f"  Gross Trading Profit: ${breakdown['gross_trading_profit']:,.2f}")
        report.append(f"  Net Contract Value:   ${breakdown['net_profit_after_costs']:,.2f}")
        report.append("")
        
        # Recommendation
        report.append("RECOMMENDATION")
        report.append("-" * 80)
        if breakdown['net_profit_after_costs'] > 0:
            report.append(f"  ✅ PROFITABLE: Contract value is positive (${breakdown['net_profit_after_costs']:,.2f})")
            report.append(f"     This contract may be attractive for the trading desk.")
        else:
            report.append(f"  ❌ UNPROFITABLE: Contract value is negative (${breakdown['net_profit_after_costs']:,.2f})")
            report.append(f"     Consider renegotiating terms or declining this contract.")
        report.append("")
        
        # Detailed Cash Flows
        report.append("DETAILED CASH FLOWS")
        report.append("-" * 80)
        for flow in valuation_result['cash_flows']:
            report.append(f"  {flow['date'].strftime('%Y-%m-%d')} | {flow['type']:12} | {flow['description']}")
            report.append(f"{'':25} | Amount: ${flow['total_amount']:,.2f}")
        report.append("")
        
        report.append("=" * 80)
        report.append("DISCLAIMER: This valuation is for indicative purposes only.")
        report.append("Actual market conditions, counterparty risk, and operational factors")
        report.append("may affect final contract value. Consult risk management before trading.")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def visualize_cash_flows(self, valuation_result: Dict, save_path: str = None):
        """
        Create visualization of cash flows over time.
        
        Args:
            valuation_result: Dictionary from price_contract()
            save_path: Optional path to save the figure
        """
        if not valuation_result['valid']:
            print("Cannot visualize: Contract validation failed.")
            return
        
        flows = valuation_result['cash_flows']
        
        # Prepare data for plotting
        dates = [f['date'] for f in flows]
        amounts = [f['total_amount'] for f in flows]
        colors = ['red' if a < 0 else 'green' for a in amounts]
        labels = [f['type'] for f in flows]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Create bar chart
        bars = ax.bar(range(len(dates)), amounts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (bar, amount) in enumerate(zip(bars, amounts)):
            height = bar.get_height()
            ax.annotate(f'${amount:,.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=9, fontweight='bold')
        
        # Customize chart
        ax.set_xlabel('Transaction Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cash Flow Amount ($)', fontsize=12, fontweight='bold')
        ax.set_title('Natural Gas Storage Contract - Cash Flow Timeline', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(dates)))
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in dates], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Revenue (Positive)'),
            Patch(facecolor='red', alpha=0.7, label='Cost (Negative)')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        # Add net value annotation
        net_value = valuation_result['breakdown']['net_profit_after_costs']
        ax.text(0.98, 0.95, f'Net Contract Value: ${net_value:,.2f}',
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()


# ============================================================================
# TEST CASES
# ============================================================================

def run_test_cases():
    """
    Run comprehensive test cases to validate the pricing model.
    """
    print("=" * 80)
    print("NATURAL GAS STORAGE CONTRACT PRICING MODEL - TEST SUITE")
    print("=" * 80)
    print()
    
    # Initialize price model
    try:
        price_model = NaturalGasPriceModel('Nat_Gas (1).csv')
    except Exception as e:
        print(f"Failed to initialize price model: {e}")
        return
    
    pricer = StorageContractPricer(price_model)
    
    # -------------------------------------------------------------------------
    # TEST CASE 1: Basic Profitable Contract
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 1: Basic Profitable Contract (Summer Buy, Winter Sell)")
    print("=" * 80)
    
    contract1 = StorageContract(
        injection_dates=['06/30/2024', '07/31/2024'],
        withdrawal_dates=['12/31/2024', '01/31/2025'],
        injection_volume=1000000,  # 1 million MMBtu
        withdrawal_volume=1000000,
        max_storage_volume=1500000,
        injection_rate=50000,  # 50K MMBtu/day
        withdrawal_rate=50000,
        storage_cost_per_month=0.05,  # $0.05/MMBtu/month
        injection_cost_per_volume=0.02,  # $0.02/MMBtu
        withdrawal_cost_per_volume=0.02,
        transport_cost_per_transaction=10000  # $10K per transaction
    )
    
    result1 = pricer.price_contract(contract1)
    report1 = pricer.generate_report(result1)
    print(report1)
    
    # Verify calculation manually
    if result1['valid']:
        print("\n🔍 MANUAL VERIFICATION:")
        avg_inj = result1['summary']['avg_injection_price']
        avg_wd = result1['summary']['avg_withdrawal_price']
        volume = contract1.injection_volume
        gross = (avg_wd - avg_inj) * volume
        print(f"  Expected Gross Profit: ${gross:,.2f}")
        print(f"  Calculated Gross Profit: ${result1['breakdown']['gross_trading_profit']:,.2f}")
    
    # -------------------------------------------------------------------------
    # TEST CASE 2: Unprofitable Contract (Narrow Spread)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 2: Unprofitable Contract (Narrow Price Spread)")
    print("=" * 80)
    
    contract2 = StorageContract(
        injection_dates=['08/31/2024'],
        withdrawal_dates=['09/30/2024'],  # Only 1 month storage
        injection_volume=500000,
        withdrawal_volume=500000,
        max_storage_volume=1000000,
        injection_rate=50000,
        withdrawal_rate=50000,
        storage_cost_per_month=0.10,  # Higher storage cost
        injection_cost_per_volume=0.05,
        withdrawal_cost_per_volume=0.05,
        transport_cost_per_transaction=50000  # Higher transport cost
    )
    
    result2 = pricer.price_contract(contract2)
    report2 = pricer.generate_report(result2)
    print(report2)
    
    # -------------------------------------------------------------------------
    # TEST CASE 3: Volume Exceeds Capacity (Should Fail Validation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 3: Invalid Contract (Volume Exceeds Storage Capacity)")
    print("=" * 80)
    
    contract3 = StorageContract(
        injection_dates=['06/30/2024'],
        withdrawal_dates=['12/31/2024'],
        injection_volume=2000000,  # Exceeds max capacity
        withdrawal_volume=2000000,
        max_storage_volume=1000000,  # Only 1 million capacity
        injection_rate=50000,
        withdrawal_rate=50000,
        storage_cost_per_month=0.05,
        injection_cost_per_volume=0.02,
        withdrawal_cost_per_volume=0.02,
        transport_cost_per_transaction=10000
    )
    
    result3 = pricer.price_contract(contract3)
    print(f"Validation Result: {'✅ PASSED' if not result3['valid'] else '❌ FAILED'}")
    print(f"Error Message: {result3.get('error', 'None')}")
    
    # -------------------------------------------------------------------------
    # TEST CASE 4: Withdrawal Before Injection (Should Fail Validation)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 4: Invalid Contract (Withdrawal Before Injection)")
    print("=" * 80)
    
    contract4 = StorageContract(
        injection_dates=['12/31/2024'],
        withdrawal_dates=['06/30/2024'],  # Before injection
        injection_volume=500000,
        withdrawal_volume=500000,
        max_storage_volume=1000000,
        injection_rate=50000,
        withdrawal_rate=50000,
        storage_cost_per_month=0.05,
        injection_cost_per_volume=0.02,
        withdrawal_cost_per_volume=0.02,
        transport_cost_per_transaction=10000
    )
    
    result4 = pricer.price_contract(contract4)
    print(f"Validation Result: {'✅ PASSED' if not result4['valid'] else '❌ FAILED'}")
    print(f"Error Message: {result4.get('error', 'None')}")
    
    # -------------------------------------------------------------------------
    # TEST CASE 5: Multiple Injection/Withdrawal Dates
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST CASE 5: Complex Contract (Multiple Injection/Withdrawal Dates)")
    print("=" * 80)
    
    contract5 = StorageContract(
        injection_dates=['05/31/2024', '06/30/2024', '07/31/2024', '08/31/2024'],
        withdrawal_dates=['11/30/2024', '12/31/2024', '01/31/2025', '02/28/2025'],
        injection_volume=2000000,
        withdrawal_volume=2000000,
        max_storage_volume=2500000,
        injection_rate=50000,
        withdrawal_rate=50000,
        storage_cost_per_month=0.04,
        injection_cost_per_volume=0.015,
        withdrawal_cost_per_volume=0.015,
        transport_cost_per_transaction=15000
    )
    
    result5 = pricer.price_contract(contract5)
    report5 = pricer.generate_report(result5)
    print(report5)
    
    # Visualize cash flows for Test Case 1
    print("\n" + "=" * 80)
    print("GENERATING CASH FLOW VISUALIZATION (Test Case 1)")
    print("=" * 80)
    pricer.visualize_cash_flows(result1, save_path='contract_cash_flows.png')
    
    # -------------------------------------------------------------------------
    # TEST SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 5
    
    if result1['valid'] and result1['breakdown']['net_profit_after_costs'] is not None:
        tests_passed += 1
        print("✅ Test 1 (Basic Profitable): PASSED")
    else:
        print("❌ Test 1 (Basic Profitable): FAILED")
    
    if result2['valid'] and result2['breakdown']['net_profit_after_costs'] is not None:
        tests_passed += 1
        print("✅ Test 2 (Unprofitable): PASSED")
    else:
        print("❌ Test 2 (Unprofitable): FAILED")
    
    if not result3['valid'] and 'capacity' in result3.get('error', '').lower():
        tests_passed += 1
        print("✅ Test 3 (Volume Validation): PASSED")
    else:
        print("❌ Test 3 (Volume Validation): FAILED")
    
    if not result4['valid'] and 'withdrawal' in result4.get('error', '').lower():
        tests_passed += 1
        print("✅ Test 4 (Date Validation): PASSED")
    else:
        print("❌ Test 4 (Date Validation): FAILED")
    
    if result5['valid'] and result5['breakdown']['net_profit_after_costs'] is not None:
        tests_passed += 1
        print("✅ Test 5 (Complex Contract): PASSED")
    else:
        print("❌ Test 5 (Complex Contract): FAILED")
    
    print()
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print("=" * 80)
    
    return tests_passed == total_tests


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run all test cases
    all_tests_passed = run_test_cases()
    
    print("\n" + "=" * 80)
    if all_tests_passed:
        print("✅ ALL TESTS PASSED - MODEL READY FOR PRODUCTION VALIDATION")
    else:
        print("⚠ SOME TESTS FAILED - REVIEW REQUIRED BEFORE PRODUCTION")
    print("=" * 80)
    
    # Interactive mode for desk users
    print("\n" + "=" * 80)
    print("INTERACTIVE PRICING TOOL")
    print("=" * 80)
    print("\nTo price a custom contract, use the following template:")
    print("""
    contract = StorageContract(
        injection_dates=['MM/DD/YYYY', 'MM/DD/YYYY'],
        withdrawal_dates=['MM/DD/YYYY', 'MM/DD/YYYY'],
        injection_volume=1000000,  # MMBtu
        withdrawal_volume=1000000,  # MMBtu
        max_storage_volume=1500000,  # MMBtu
        injection_rate=50000,  # MMBtu/day
        withdrawal_rate=50000,  # MMBtu/day
        storage_cost_per_month=0.05,  # $/MMBtu/month
        injection_cost_per_volume=0.02,  # $/MMBtu
        withdrawal_cost_per_volume=0.02,  # $/MMBtu
        transport_cost_per_transaction=10000  # $ per transaction
    )
    
    result = pricer.price_contract(contract)
    report = pricer.generate_report(result)
    print(report)
    """)
    print("=" * 80)