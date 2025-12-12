"""
Data Preprocessing Module for Credit Score Prediction
Based on the original notebook preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import re
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns."""
    cols_to_drop = ['ID', 'Month', 'Name', 'Number']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols)
    logger.info(f"Dropped columns: {existing_cols}")
    return df


def clean_dirty(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dirty values in the dataset."""
    # Remove rows with Customer_ID = 0
    if 'Customer_ID' in df.columns:
        cust_0 = df[df['Customer_ID'] == 0].index
        df = df.drop(labels=cust_0, axis=0)
    
    # Clean Age column
    if 'Age' in df.columns:
        df['Age'] = df['Age'].astype(str).str.replace('_', '').astype(int)
    
    # Clean Income_Annual
    if 'Income_Annual' in df.columns:
        df['Income_Annual'] = df['Income_Annual'].astype(str).str.replace('_', '').astype(float)
    
    # Handle negative bank accounts
    if 'Total_Bank_Accounts' in df.columns:
        bank_less0 = df[df['Total_Bank_Accounts'] < 0].index
        df = df.drop(labels=bank_less0, axis=0)
    
    # Clean Total_Delayed_Payments
    if 'Total_Delayed_Payments' in df.columns:
        df['Total_Delayed_Payments'] = df['Total_Delayed_Payments'].astype(str).str.replace('_', '').astype(float)
    
    # Clean Total_Current_Loans
    if 'Total_Current_Loans' in df.columns:
        df['Total_Current_Loans'] = df['Total_Current_Loans'].astype(str).str.replace('_', '').astype(float)
    
    # Clean Credit_Limit
    if 'Credit_Limit' in df.columns:
        df['Credit_Limit'] = df['Credit_Limit'].astype(str).str.replace('_', '')
        df['Credit_Limit'] = df['Credit_Limit'].replace('', np.nan).astype(float)
    
    # Clean Credit_Mix
    if 'Credit_Mix' in df.columns:
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
    
    # Clean Current_Debt_Outstanding
    if 'Current_Debt_Outstanding' in df.columns:
        df['Current_Debt_Outstanding'] = df['Current_Debt_Outstanding'].astype(str).str.replace('_', '').astype(float)
    
    # Clean Monthly_Investment
    if 'Monthly_Investment' in df.columns:
        df['Monthly_Investment'] = df['Monthly_Investment'].astype(str).str.replace('_', '').astype(float)
    
    # Clean Payment_Behaviour
    if 'Payment_Behaviour' in df.columns:
        dirty_mapper = {'!@9#%8': np.nan}
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace(dirty_mapper)
    
    # Clean Monthly_Balance
    if 'Monthly_Balance' in df.columns:
        df['Monthly_Balance'] = df['Monthly_Balance'].astype(str).str.replace('_', '').astype(float)
    
    # Clean Profession
    if 'Profession' in df.columns:
        df['Profession'] = df['Profession'].replace('_______', 'Other')
    
    logger.info("Cleaned dirty values")
    return df


def ohe_loan_types(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode loan types."""
    if 'Loan_Type' not in df.columns:
        return df
    
    df['Loan_Type'] = df['Loan_Type'].astype(str)
    unique_loan_types = df['Loan_Type'].str.split(',').explode().str.replace('and', '').str.strip().unique()
    
    for loan_type in unique_loan_types:
        cleaned_loan_type = loan_type.replace(' ', '_').replace('-', '_').lower()
        df[cleaned_loan_type] = df['Loan_Type'].apply(lambda x, lt=loan_type: x.count(lt))
    
    df = df.rename(columns={'nan': 'loan_not_taken'})
    
    cols_to_drop = ['Loan_Type']
    if 'Total_Current_Loans' in df.columns:
        cols_to_drop.append('Total_Current_Loans')
    
    df = df.drop(columns=cols_to_drop)
    logger.info("One-hot encoded loan types")
    return df


def convert_credit_history_to_months(df: pd.DataFrame) -> pd.DataFrame:
    """Convert credit history age to months."""
    if 'Credit_History_Age' not in df.columns:
        return df
    
    def convert_to_months(value):
        if pd.isnull(value):
            return value
        match = re.match(r"(\d+)\s+Years\s+and\s+(\d+)\s+Months", str(value))
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return years * 12 + months
        return None
    
    df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_to_months)
    logger.info("Converted credit history to months")
    return df


def ohe_profession(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode profession."""
    if 'Profession' in df.columns:
        df = pd.get_dummies(df, columns=['Profession'])
        logger.info("One-hot encoded profession")
    return df


def ohe_payment(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode payment behaviour."""
    if 'Payment_Behaviour' in df.columns:
        df = pd.get_dummies(df, columns=['Payment_Behaviour'])
        logger.info("One-hot encoded payment behaviour")
    return df


def ohe_payment_min_amt(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode payment of minimum amount."""
    if 'Payment_of_Min_Amount' in df.columns:
        df = pd.get_dummies(df, columns=['Payment_of_Min_Amount'])
        logger.info("One-hot encoded payment of minimum amount")
    return df


def fill_na_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NA values per Customer_ID."""
    if 'Customer_ID' not in df.columns:
        # If no Customer_ID, fill with overall median/mode
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        if 'Credit_Mix' in df.columns:
            mode_val = df['Credit_Mix'].mode()
            if not mode_val.empty:
                df['Credit_Mix'] = df['Credit_Mix'].fillna(mode_val[0])
        return df
    
    unique_customers = df['Customer_ID'].unique()
    
    for i in unique_customers:
        data = df[df['Customer_ID'] == i]
        data_numeric_cols = data.select_dtypes(include='number').columns
        out = data.fillna(data[data_numeric_cols].median())
        
        if 'Credit_Mix' in out.columns:
            if not out['Credit_Mix'].mode().empty:
                mode_value = out['Credit_Mix'].mode()[0]
                out['Credit_Mix'] = out['Credit_Mix'].fillna(mode_value)
            else:
                out = out.dropna(subset=['Credit_Mix'])
        
        df.loc[df['Customer_ID'] == i, :] = out
    
    df = df.drop(columns='Customer_ID')
    logger.info("Filled NA values")
    return df


def label_encode_credit_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode credit mix."""
    if 'Credit_Mix' in df.columns:
        mapper = {'Bad': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Mix'] = df['Credit_Mix'].map(mapper)
        logger.info("Label encoded credit mix")
    return df


def le_target(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode target variable."""
    if 'Credit_Score' in df.columns:
        mapper = {'Poor': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Score'] = df['Credit_Score'].map(mapper)
        logger.info("Label encoded target")
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cap outliers using IQR method."""
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound,
                          np.where(df[col] > upper_bound, upper_bound, df[col]))
    
    logger.info(f"Capped outliers for {len(columns)} columns")
    return df


def preprocess(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Main preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        is_training: Whether this is training data (has Credit_Score column)
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Starting preprocessing pipeline (training={is_training})")
    
    df = clean_dirty(df)
    df = drop_cols(df)
    df = convert_credit_history_to_months(df)
    df = fill_na_vals(df)
    df = ohe_loan_types(df)
    df = ohe_payment(df)
    df = ohe_profession(df)
    df = label_encode_credit_mix(df)
    df = ohe_payment_min_amt(df)
    
    if is_training:
        df = le_target(df)
        df = df.dropna()
    
    df = df.reset_index(drop=True)
    
    # Cap outliers for numeric columns
    numeric_cols = ['Age', 'Income_Annual', 'Base_Salary_PerMonth', 'Total_Bank_Accounts',
                   'Total_Credit_Cards', 'Rate_Of_Interest', 'Delay_from_due_date',
                   'Total_Delayed_Payments', 'Credit_Limit', 'Total_Credit_Enquiries',
                   'Current_Debt_Outstanding', 'Ratio_Credit_Utilization',
                   'Credit_History_Age', 'Per_Month_EMI', 'Monthly_Investment',
                   'Monthly_Balance']
    
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    df = cap_outliers_iqr(df, existing_numeric_cols)
    
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df


def align_columns(train_cols: list, test_df: pd.DataFrame) -> pd.DataFrame:
    """Align test dataframe columns with training columns."""
    # Add missing columns with zeros
    for col in train_cols:
        if col not in test_df.columns:
            test_df[col] = 0
    
    # Keep only training columns (in same order)
    test_df = test_df[train_cols]
    return test_df
