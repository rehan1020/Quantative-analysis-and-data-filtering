import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
import joblib
from typing import Dict, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class LoanDataProcessor:
    """Handles data loading, cleaning, and feature engineering"""
    
    def __init__(self, file_path: str = 'Task 3 and 4_Loan_Data.csv'):
        self.file_path = file_path
        self.df = None
        self.feature_columns = None
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load and inspect the loan data"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✓ Data loaded successfully: {len(self.df)} records")
            return self.df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            raise
    
    def inspect_data(self) -> Dict:
        """Provide comprehensive data inspection"""
        inspection = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'default_distribution': self.df['default'].value_counts().to_dict(),
            'default_rate': self.df['default'].mean()
        }
        
        print("\n" + "="*70)
        print("DATA INSPECTION REPORT")
        print("="*70)
        print(f"Records: {inspection['shape'][0]}, Features: {inspection['shape'][1]}")
        print(f"Default Rate: {inspection['default_rate']:.2%}")
        print(f"Missing Values: {sum(inspection['missing_values'].values())}")
        print("="*70)
        
        return inspection
    
    def create_features(self) -> pd.DataFrame:
        """Create additional predictive features"""
        df = self.df.copy()
        
        # Debt-to-Income Ratio
        df['debt_to_income'] = df['total_debt_outstanding'] / (df['income'] + 1)
        
        # Credit Utilization
        df['credit_utilization'] = df['loan_amt_outstanding'] / (df['credit_lines_outstanding'] + 1)
        
        # Loan-to-Income Ratio
        df['loan_to_income'] = df['loan_amt_outstanding'] / (df['income'] + 1)
        
        # FICO Score Categories
        df['fico_category'] = pd.cut(
            df['fico_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        df['fico_category_encoded'] = df['fico_category'].cat.codes
        
        # Employment Stability
        df['employment_stability'] = df['years_employed'] / (df['years_employed'] + 1)
        
        # Total Debt Ratio
        df['total_debt_ratio'] = df['total_debt_outstanding'] / (df['income'] + 1)
        
        print(f"✓ Created {6} new features")
        return df
    
    def prepare_features(self, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""
        if df is None:
            df = self.create_features()
        
        # Select feature columns (exclude target and identifiers)
        self.feature_columns = [
            'credit_lines_outstanding',
            'loan_amt_outstanding',
            'total_debt_outstanding',
            'income',
            'years_employed',
            'fico_score',
            'debt_to_income',
            'credit_utilization',
            'loan_to_income',
            'fico_category_encoded',
            'employment_stability',
            'total_debt_ratio'
        ]
        
        X = df[self.feature_columns].copy()
        y = df['default'].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_columns
        )
        
        print(f"✓ Prepared {len(self.feature_columns)} features for modeling")
        return X_scaled, y
    
    def get_feature_columns(self) -> list:
        """Return the list of feature columns used in modeling"""
        return self.feature_columns


# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

class DefaultPredictionModel:
    """Train and evaluate multiple default prediction models"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models and compare performance"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("\n" + "="*70)
        print("MODEL TRAINING & EVALUATION")
        print("="*70)
        
        # Define models to compare
        model_configs = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        }
        
        # Train and evaluate each model
        for name, model in model_configs.items():
            print(f"\nTraining {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_pred_proba),
                'model': model,
                'y_pred_proba': y_pred_proba
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"  AUC-ROC: {metrics['auc_roc']:.4f} (CV: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f})")
            print(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Select best model based on AUC-ROC
        self.best_model_name = max(self.results, key=lambda x: self.results[x]['auc_roc'])
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*70)
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"Test AUC-ROC: {self.results[self.best_model_name]['auc_roc']:.4f}")
        print("="*70)
        
        return self.results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.models['Random Forest'].feature_importances_ 
                    if 'Random Forest' in self.models 
                    else list(range(len(self.best_model.coef_[0]))),
                'importance': self.best_model.feature_importances_ 
                    if hasattr(self.best_model, 'feature_importances_')
                    else np.abs(self.best_model.coef_[0])
            }).sort_values('importance', ascending=False)
            return importance
        return None
    
    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probability using the best model"""
        return self.best_model.predict_proba(X)[:, 1]
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, metrics in self.results.items():
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {metrics["auc_roc"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Plot confusion matrix for best model"""
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXPECTED LOSS CALCULATOR
# ============================================================================

class ExpectedLossCalculator:
    """Calculate expected loss for individual loans or portfolios"""
    
    def __init__(self, data_processor: LoanDataProcessor, 
                 prediction_model: DefaultPredictionModel,
                 recovery_rate: float = 0.10):
        self.data_processor = data_processor
        self.prediction_model = prediction_model
        self.recovery_rate = recovery_rate
        self.lgd = 1 - recovery_rate  # Loss Given Default
        
    def calculate_expected_loss(self, loan_data: Union[Dict, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        """
        Calculate expected loss for a loan or portfolio
        
        Expected Loss = PD × LGD × EAD
        
        Parameters:
        -----------
        loan_data : dict or DataFrame
            Loan properties including:
            - credit_lines_outstanding
            - loan_amt_outstanding
            - total_debt_outstanding
            - income
            - years_employed
            - fico_score
        
        Returns:
        --------
        float or DataFrame : Expected loss amount
        """
        
        # Handle single loan (dict)
        if isinstance(loan_data, dict):
            loan_df = pd.DataFrame([loan_data])
            single_loan = True
        else:
            loan_df = loan_data.copy()
            single_loan = False
        
        # Create features
        loan_df = self._add_features(loan_df)
        
        # Prepare features for prediction
        X = loan_df[self.data_processor.get_feature_columns()].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.data_processor.scaler.transform(X),
            columns=self.data_processor.get_feature_columns()
        )
        
        # Predict PD
        pd_values = self.prediction_model.predict_probability(X_scaled)
        
        # Calculate Expected Loss
        # EAD = loan_amt_outstanding
        ead = loan_df['loan_amt_outstanding'].values
        expected_loss = pd_values * self.lgd * ead
        
        # Add to dataframe
        loan_df['probability_of_default'] = pd_values
        loan_df['expected_loss'] = expected_loss
        loan_df['loss_given_default'] = self.lgd
        loan_df['exposure_at_default'] = ead
        
        if single_loan:
            return {
                'probability_of_default': float(pd_values[0]),
                'expected_loss': float(expected_loss[0]),
                'loss_given_default': self.lgd,
                'exposure_at_default': float(ead[0]),
                'loan_amount': float(ead[0])
            }
        
        return loan_df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to loan data"""
        df = df.copy()
        
        # Debt-to-Income Ratio
        df['debt_to_income'] = df['total_debt_outstanding'] / (df['income'] + 1)
        
        # Credit Utilization
        df['credit_utilization'] = df['loan_amt_outstanding'] / (df['credit_lines_outstanding'] + 1)
        
        # Loan-to-Income Ratio
        df['loan_to_income'] = df['loan_amt_outstanding'] / (df['income'] + 1)
        
        # FICO Category (simplified for single predictions)
        df['fico_category_encoded'] = pd.cut(
            df['fico_score'],
            bins=[0, 580, 670, 740, 850],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        # Employment Stability
        df['employment_stability'] = df['years_employed'] / (df['years_employed'] + 1)
        
        # Total Debt Ratio
        df['total_debt_ratio'] = df['total_debt_outstanding'] / (df['income'] + 1)
        
        return df
    
    def calculate_portfolio_loss(self, loan_portfolio: pd.DataFrame) -> Dict:
        """Calculate aggregate loss metrics for a loan portfolio"""
        
        results = self.calculate_expected_loss(loan_portfolio)
        
        portfolio_metrics = {
            'total_exposure': results['exposure_at_default'].sum(),
            'total_expected_loss': results['expected_loss'].sum(),
            'average_pd': results['probability_of_default'].mean(),
            'weighted_average_pd': (
                results['probability_of_default'] * results['exposure_at_default']
            ).sum() / results['exposure_at_default'].sum(),
            'expected_loss_rate': (
                results['expected_loss'].sum() / results['exposure_at_default'].sum()
            ),
            'number_of_loans': len(results),
            'high_risk_loans': (results['probability_of_default'] > 0.3).sum(),
            'recovery_rate': self.recovery_rate
        }
        
        return portfolio_metrics
    
    def generate_loss_report(self, loan_portfolio: pd.DataFrame) -> str:
        """Generate a comprehensive loss report"""
        
        metrics = self.calculate_portfolio_loss(loan_portfolio)
        
        report = []
        report.append("=" * 70)
        report.append("LOAN PORTFOLIO EXPECTED LOSS REPORT")
        report.append("=" * 70)
        report.append("")
        report.append("PORTFOLIO SUMMARY")
        report.append("-" * 70)
        report.append(f"Total Number of Loans:        {metrics['number_of_loans']:,}")
        report.append(f"Total Exposure (EAD):         ${metrics['total_exposure']:,.2f}")
        report.append(f"Total Expected Loss:          ${metrics['total_expected_loss']:,.2f}")
        report.append(f"Expected Loss Rate:           {metrics['expected_loss_rate']:.2%}")
        report.append("")
        report.append("RISK METRICS")
        report.append("-" * 70)
        report.append(f"Average Probability of Default: {metrics['average_pd']:.2%}")
        report.append(f"Weighted Average PD:            {metrics['weighted_average_pd']:.2%}")
        report.append(f"High Risk Loans (PD > 30%):     {metrics['high_risk_loans']:,} ({metrics['high_risk_loans']/metrics['number_of_loans']:.2%})")
        report.append("")
        report.append("ASSUMPTIONS")
        report.append("-" * 70)
        report.append(f"Recovery Rate:                  {metrics['recovery_rate']:.0%}")
        report.append(f"Loss Given Default (LGD):       {self.lgd:.0%}")
        report.append("")
        report.append("CAPITAL ADEQUACY RECOMMENDATION")
        report.append("-" * 70)
        report.append(f"Minimum Capital Reserve:        ${metrics['total_expected_loss']:,.2f}")
        report.append(f"Recommended Reserve (2x EL):    ${metrics['total_expected_loss'] * 2:,.2f}")
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


# ============================================================================
# COMPREHENSIVE TESTING SUITE
# ============================================================================

class ModelTestSuite:
    """Comprehensive testing for the loan default prediction system"""
    
    def __init__(self, data_processor: LoanDataProcessor, 
                 prediction_model: DefaultPredictionModel,
                 loss_calculator: ExpectedLossCalculator):
        self.data_processor = data_processor
        self.prediction_model = prediction_model
        self.loss_calculator = loss_calculator
        self.test_results = {}
        
    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE MODEL TESTING SUITE")
        print("="*70)
        
        tests = [
            ('Data Loading Test', self.test_data_loading),
            ('Feature Engineering Test', self.test_feature_engineering),
            ('Model Prediction Test', self.test_model_prediction),
            ('Expected Loss Calculation Test', self.test_expected_loss),
            ('Portfolio Loss Test', self.test_portfolio_loss),
            ('Edge Cases Test', self.test_edge_cases),
            ('Model Performance Test', self.test_model_performance)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print('='*70)
            try:
                result = test_func()
                self.test_results[test_name] = {'status': 'PASS', 'result': result}
                print(f"✓ {test_name}: PASSED")
            except Exception as e:
                self.test_results[test_name] = {'status': 'FAIL', 'error': str(e)}
                print(f"✗ {test_name}: FAILED - {e}")
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'PASS')
        total = len(self.test_results)
        print(f"Tests Passed: {passed}/{total}")
        
        for name, result in self.test_results.items():
            status = "✓ PASS" if result['status'] == 'PASS' else f"✗ FAIL: {result.get('error', '')}"
            print(f"  {name}: {status}")
        
        print("="*70)
        
        return self.test_results
    
    def test_data_loading(self) -> bool:
        """Test data loading and basic validation"""
        df = self.data_processor.df
        
        # Check required columns
        required_cols = ['customer_id', 'income', 'fico_score', 'default', 'loan_amt_outstanding']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check data types
        assert df['default'].isin([0, 1]).all(), "Default column must be binary (0 or 1)"
        assert (df['income'] >= 0).all(), "Income must be non-negative"
        assert (df['fico_score'] >= 300).all(), "FICO score must be >= 300"
        assert (df['fico_score'] <= 850).all(), "FICO score must be <= 850"
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        assert missing_pct < 0.1, f"Too many missing values: {missing_pct:.2%}"
        
        print(f"  Records: {len(df)}")
        print(f"  Default Rate: {df['default'].mean():.2%}")
        print(f"  Missing Values: {missing_pct:.2%}")
        
        return True
    
    def test_feature_engineering(self) -> bool:
        """Test feature engineering pipeline"""
        df_features = self.data_processor.create_features()
        
        # Check new features exist
        new_features = ['debt_to_income', 'credit_utilization', 'loan_to_income',
                       'fico_category_encoded', 'employment_stability', 'total_debt_ratio']
        
        for feature in new_features:
            assert feature in df_features.columns, f"Missing feature: {feature}"
        
        # Check feature ranges
        assert (df_features['debt_to_income'] >= 0).all(), "debt_to_income must be non-negative"
        assert (df_features['fico_category_encoded'] >= 0).all(), "fico_category_encoded must be non-negative"
        assert (df_features['employment_stability'] >= 0).all(), "employment_stability must be non-negative"
        assert (df_features['employment_stability'] <= 1).all(), "employment_stability must be <= 1"
        
        print(f"  Features Created: {len(new_features)}")
        print(f"  Total Features: {len(df_features.columns)}")
        
        return True
    
    def test_model_prediction(self) -> bool:
        """Test model prediction functionality"""
        X, y = self.data_processor.prepare_features()
        
        # Split for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Test predictions
        probabilities = self.prediction_model.predict_probability(X_test)
        
        # Validate probabilities
        assert len(probabilities) == len(X_test), "Prediction count mismatch"
        assert (probabilities >= 0).all(), "Probabilities must be >= 0"
        assert (probabilities <= 1).all(), "Probabilities must be <= 1"
        
        # Check probability distribution
        print(f"  Mean PD: {probabilities.mean():.4f}")
        print(f"  Std PD: {probabilities.std():.4f}")
        print(f"  Min PD: {probabilities.min():.4f}")
        print(f"  Max PD: {probabilities.max():.4f}")
        
        return True
    
    def test_expected_loss(self) -> bool:
        """Test expected loss calculation for single loan"""
        
        # Test loan
        test_loan = {
            'credit_lines_outstanding': 3,
            'loan_amt_outstanding': 50000,
            'total_debt_outstanding': 75000,
            'income': 80000,
            'years_employed': 5,
            'fico_score': 650
        }
        
        result = self.loss_calculator.calculate_expected_loss(test_loan)
        
        # Validate results
        assert 'probability_of_default' in result, "Missing PD in result"
        assert 'expected_loss' in result, "Missing expected loss in result"
        assert 'exposure_at_default' in result, "Missing EAD in result"
        
        assert result['probability_of_default'] >= 0, "PD must be >= 0"
        assert result['probability_of_default'] <= 1, "PD must be <= 1"
        assert result['expected_loss'] >= 0, "Expected loss must be >= 0"
        assert result['exposure_at_default'] == 50000, "EAD should match loan amount"
        
        # Verify EL = PD × LGD × EAD
        expected_el = result['probability_of_default'] * self.loss_calculator.lgd * result['exposure_at_default']
        assert abs(result['expected_loss'] - expected_el) < 0.01, "EL calculation error"
        
        print(f"  Loan Amount: ${result['exposure_at_default']:,.2f}")
        print(f"  Probability of Default: {result['probability_of_default']:.2%}")
        print(f"  Loss Given Default: {result['loss_given_default']:.0%}")
        print(f"  Expected Loss: ${result['expected_loss']:,.2f}")
        
        return True
    
    def test_portfolio_loss(self) -> bool:
        """Test portfolio-level loss calculation"""
        
        # Create test portfolio
        test_portfolio = pd.DataFrame([
            {
                'credit_lines_outstanding': 3,
                'loan_amt_outstanding': 50000,
                'total_debt_outstanding': 75000,
                'income': 80000,
                'years_employed': 5,
                'fico_score': 650
            },
            {
                'credit_lines_outstanding': 5,
                'loan_amt_outstanding': 100000,
                'total_debt_outstanding': 150000,
                'income': 120000,
                'years_employed': 10,
                'fico_score': 720
            },
            {
                'credit_lines_outstanding': 1,
                'loan_amt_outstanding': 25000,
                'total_debt_outstanding': 40000,
                'income': 45000,
                'years_employed': 2,
                'fico_score': 580
            }
        ])
        
        metrics = self.loss_calculator.calculate_portfolio_loss(test_portfolio)
        
        # Validate metrics
        assert metrics['number_of_loans'] == 3, "Loan count mismatch"
        assert metrics['total_exposure'] == 175000, "Total exposure mismatch"
        assert metrics['total_expected_loss'] >= 0, "Expected loss must be >= 0"
        assert metrics['average_pd'] >= 0, "Average PD must be >= 0"
        assert metrics['average_pd'] <= 1, "Average PD must be <= 1"
        
        print(f"  Portfolio Size: {metrics['number_of_loans']} loans")
        print(f"  Total Exposure: ${metrics['total_exposure']:,.2f}")
        print(f"  Total Expected Loss: ${metrics['total_expected_loss']:,.2f}")
        print(f"  Average PD: {metrics['average_pd']:.2%}")
        print(f"  Expected Loss Rate: {metrics['expected_loss_rate']:.2%}")
        
        return True
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and boundary conditions"""
        
        # Test 1: Very low FICO score
        low_fico_loan = {
            'credit_lines_outstanding': 1,
            'loan_amt_outstanding': 10000,
            'total_debt_outstanding': 15000,
            'income': 30000,
            'years_employed': 1,
            'fico_score': 300  # Minimum FICO
        }
        result_low = self.loss_calculator.calculate_expected_loss(low_fico_loan)
        assert result_low['probability_of_default'] >= 0, "Low FICO PD invalid"
        print(f"  Low FICO (300) PD: {result_low['probability_of_default']:.2%}")
        
        # Test 2: Very high FICO score
        high_fico_loan = {
            'credit_lines_outstanding': 10,
            'loan_amt_outstanding': 200000,
            'total_debt_outstanding': 250000,
            'income': 250000,
            'years_employed': 20,
            'fico_score': 850  # Maximum FICO
        }
        result_high = self.loss_calculator.calculate_expected_loss(high_fico_loan)
        assert result_high['probability_of_default'] >= 0, "High FICO PD invalid"
        print(f"  High FICO (850) PD: {result_high['probability_of_default']:.2%}")
        
        # Test 3: Zero income (edge case)
        zero_income_loan = {
            'credit_lines_outstanding': 2,
            'loan_amt_outstanding': 30000,
            'total_debt_outstanding': 50000,
            'income': 0,
            'years_employed': 0,
            'fico_score': 600
        }
        result_zero = self.loss_calculator.calculate_expected_loss(zero_income_loan)
        assert result_zero['probability_of_default'] >= 0, "Zero income PD invalid"
        print(f"  Zero Income PD: {result_zero['probability_of_default']:.2%}")
        
        # Test 4: Very large loan
        large_loan = {
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 1000000,
            'total_debt_outstanding': 1500000,
            'income': 500000,
            'years_employed': 15,
            'fico_score': 700
        }
        result_large = self.loss_calculator.calculate_expected_loss(large_loan)
        assert result_large['expected_loss'] > 0, "Large loan EL should be > 0"
        print(f"  Large Loan ($1M) EL: ${result_large['expected_loss']:,.2f}")
        
        # Verify FICO relationship (lower FICO should have higher PD)
        assert result_low['probability_of_default'] >= result_high['probability_of_default'], \
            "Low FICO should have higher PD than high FICO"
        
        print("  ✓ All edge cases handled correctly")
        
        return True
    
    def test_model_performance(self) -> bool:
        """Test model performance metrics"""
        
        results = self.prediction_model.results
        
        # Check all models were trained
        assert len(results) >= 2, "Should have at least 2 models"
        
        # Check performance thresholds
        best_auc = max(r['auc_roc'] for r in results.values())
        assert best_auc >= 0.7, f"Best model AUC ({best_auc:.4f}) below threshold (0.7)"
        
        # Check cross-validation consistency
        for name, metrics in results.items():
            cv_mean = metrics['cv_auc_mean']
            test_auc = metrics['auc_roc']
            cv_std = metrics['cv_auc_std']
            
            # CV and test should be reasonably close
            assert abs(cv_mean - test_auc) < 0.1, \
                f"{name}: CV-Test gap too large ({cv_mean:.4f} vs {test_auc:.4f})"
            
            print(f"  {name}:")
            print(f"    Test AUC: {test_auc:.4f}")
            print(f"    CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
        
        print("  ✓ Model performance meets standards")
        
        return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("LOAN DEFAULT PREDICTION & EXPECTED LOSS MODEL")
    print("="*70)
    
    # Step 1: Load and process data
    print("\n[STEP 1] Loading and Processing Data...")
    data_processor = LoanDataProcessor('Task 3 and 4_Loan_Data.csv')
    data_processor.load_data()
    data_processor.inspect_data()
    
    # Step 2: Prepare features
    print("\n[STEP 2] Preparing Features...")
    X, y = data_processor.prepare_features()
    
    # Step 3: Train models
    print("\n[STEP 3] Training Prediction Models...")
    prediction_model = DefaultPredictionModel()
    prediction_model.train_models(X, y)
    
    # Step 4: Create loss calculator
    print("\n[STEP 4] Initializing Expected Loss Calculator...")
    loss_calculator = ExpectedLossCalculator(
        data_processor=data_processor,
        prediction_model=prediction_model,
        recovery_rate=0.10
    )
    
    # Step 5: Run comprehensive tests
    print("\n[STEP 5] Running Test Suite...")
    test_suite = ModelTestSuite(data_processor, prediction_model, loss_calculator)
    test_results = test_suite.run_all_tests()
    
    # Step 6: Generate sample report
    print("\n[STEP 6] Generating Sample Loss Report...")
    sample_portfolio = data_processor.df.head(100)  # First 100 loans
    report = loss_calculator.generate_loss_report(sample_portfolio)
    print("\n" + report)
    
    # Step 7: Demonstrate single loan calculation
    print("\n[STEP 7] Single Loan Expected Loss Example...")
    example_loan = {
        'credit_lines_outstanding': 3,
        'loan_amt_outstanding': 50000,
        'total_debt_outstanding': 75000,
        'income': 80000,
        'years_employed': 5,
        'fico_score': 650
    }
    
    loan_result = loss_calculator.calculate_expected_loss(example_loan)
    print("\nExample Loan Details:")
    print(f"  Loan Amount: ${loan_result['exposure_at_default']:,.2f}")
    print(f"  FICO Score: {example_loan['fico_score']}")
    print(f"  Income: ${example_loan['income']:,}")
    print(f"  Years Employed: {example_loan['years_employed']}")
    print(f"\nRisk Assessment:")
    print(f"  Probability of Default: {loan_result['probability_of_default']:.2%}")
    print(f"  Loss Given Default: {loan_result['loss_given_default']:.0%}")
    print(f"  Expected Loss: ${loan_result['expected_loss']:,.2f}")
    
    # Step 8: Feature importance
    print("\n[STEP 8] Feature Importance Analysis...")
    feature_importance = prediction_model.get_feature_importance()
    if feature_importance is not None:
        print("\nTop 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Final summary
    print("\n" + "="*70)
    print("MODEL DEPLOYMENT READY")
    print("="*70)
    print(f"Best Model: {prediction_model.best_model_name}")
    print(f"Test AUC-ROC: {prediction_model.results[prediction_model.best_model_name]['auc_roc']:.4f}")
    print(f"All Tests Passed: {all(r['status'] == 'PASS' for r in test_results.values())}")
    print("="*70)
    
    return {
        'data_processor': data_processor,
        'prediction_model': prediction_model,
        'loss_calculator': loss_calculator,
        'test_results': test_results
    }


# ============================================================================
# PRODUCTION-READY FUNCTION
# ============================================================================

def calculate_loan_expected_loss(
    credit_lines_outstanding: float,
    loan_amt_outstanding: float,
    total_debt_outstanding: float,
    income: float,
    years_employed: float,
    fico_score: float,
    recovery_rate: float = 0.10
) -> Dict:
    """
    PRODUCTION FUNCTION: Calculate expected loss for a single loan
    
    This is the main function to be used in production. It encapsulates
    the entire pipeline from feature engineering to loss calculation.
    
    Parameters:
    -----------
    credit_lines_outstanding : float
        Number of credit lines the borrower has
    loan_amt_outstanding : float
        Current loan amount outstanding (EAD)
    total_debt_outstanding : float
        Total debt outstanding across all obligations
    income : float
        Annual income of the borrower
    years_employed : float
        Number of years at current employer
    fico_score : float
        FICO credit score (300-850)
    recovery_rate : float, default=0.10
        Expected recovery rate in case of default (10%)
    
    Returns:
    --------
    dict : Contains PD, LGD, EAD, and Expected Loss
    
    Example:
    --------
    >>> result = calculate_loan_expected_loss(
    ...     credit_lines_outstanding=3,
    ...     loan_amt_outstanding=50000,
    ...     total_debt_outstanding=75000,
    ...     income=80000,
    ...     years_employed=5,
    ...     fico_score=650
    ... )
    >>> print(f"Expected Loss: ${result['expected_loss']:,.2f}")
    """
    
    # Initialize components (in production, these would be pre-loaded)
    data_processor = LoanDataProcessor('Task 3 and 4_Loan_Data.csv')
    data_processor.load_data()
    data_processor.prepare_features()
    
    prediction_model = DefaultPredictionModel()
    prediction_model.train_models(*data_processor.prepare_features())
    
    loss_calculator = ExpectedLossCalculator(
        data_processor=data_processor,
        prediction_model=prediction_model,
        recovery_rate=recovery_rate
    )
    
    # Create loan dictionary
    loan_data = {
        'credit_lines_outstanding': credit_lines_outstanding,
        'loan_amt_outstanding': loan_amt_outstanding,
        'total_debt_outstanding': total_debt_outstanding,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score
    }
    
    # Calculate expected loss
    result = loss_calculator.calculate_expected_loss(loan_data)
    
    return result


if __name__ == "__main__":
    # Run the complete pipeline
    results = main()
    
    # Optional: Save models for production use
    # joblib.dump(results['prediction_model'].best_model, 'best_default_model.pkl')
    # joblib.dump(results['data_processor'].scaler, 'feature_scaler.pkl')