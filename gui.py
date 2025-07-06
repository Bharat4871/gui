import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import json # For parsing API response
import requests # For making API calls

# --- Configuration and Constants ---
# Set page configuration for a wide layout and a descriptive title
st.set_page_config(layout="wide", page_title="Enhanced Investment Analysis Dashboard with AI")

# Define a base starting value for historical data for easier CAGR calculation
BASE_STARTING_VALUE = 1000
# Define the range of years for historical data simulation
HISTORICAL_START_YEAR = 2001
HISTORICAL_END_YEAR = 2023

# Gemini API Key (leave empty, Canvas will inject it)
GEMINI_API_KEY = "" # If you want to use models other than gemini-2.0-flash or imagen-3.0-generate-002, provide an API key here. Otherwise, leave this as-is.


# --- Helper Functions ---

def calculate_cagr(initial_value, final_value, years):
    """
    Calculates the Compound Annual Growth Rate (CAGR).

    Args:
        initial_value (float): The starting value of the investment.
        final_value (float): The ending value of the investment.
        years (int): The number of years the investment was held.

    Returns:
        float: The CAGR as a percentage. Returns 0 if years is 0 or initial_value is 0.
    """
    if years <= 0 or initial_value <= 0:
        return 0.0
    try:
        cagr = (final_value / initial_value)**(1 / years) - 1
        return cagr * 100  # Return as percentage
    except ZeroDivisionError:
        return 0.0
    except Exception as e:
        st.error(f"Error calculating CAGR: {e}")
        return 0.0

def calculate_sip_future_value(sip_amount_monthly, annual_return_rate, investment_years, inflation_rate=0, ulip_charges=None):
    """
    Calculates the future value of a SIP investment, considering ULIP charges and inflation.

    Args:
        sip_amount_monthly (float): Monthly SIP amount.
        annual_return_rate (float): Annual return rate (e.g., 0.12 for 12%).
        investment_years (int): Total years of investment.
        inflation_rate (float): Annual inflation rate (e.g., 0.05 for 5%).
        ulip_charges (dict): Dictionary of ULIP charges (premium_allocation, policy_admin_monthly_perc, fund_management_perc, mortality_per_lakh_pa)

    Returns:
        tuple: (Future value nominal, Future value real, Total charges deducted, Total invested)
    """
    if sip_amount_monthly <= 0 or investment_years <= 0:
        return 0.0, 0.0, 0.0, 0.0

    monthly_return_rate = (1 + annual_return_rate)**(1/12) - 1
    total_months = investment_years * 12
    
    total_invested_amount = 0
    current_corpus = 0
    total_charges_deducted = 0

    for month in range(total_months):
        # Apply premium allocation charge first
        premium_after_allocation = sip_amount_monthly * (1 - ulip_charges.get('premium_allocation', 0) / 100) if ulip_charges else sip_amount_monthly
        total_invested_amount += sip_amount_monthly # Track gross investment

        # Add premium to corpus
        current_corpus += premium_after_allocation

        # Apply monthly policy administration charge (as % of of initial premium, simplified)
        if ulip_charges and ulip_charges.get('policy_admin_monthly_perc', 0) > 0:
            policy_admin_charge = sip_amount_monthly * (ulip_charges['policy_admin_monthly_perc'] / 100)
            current_corpus -= policy_admin_charge
            total_charges_deducted += policy_admin_charge

        # Apply monthly fund management charge (as % of corpus)
        if ulip_charges and ulip_charges.get('fund_management_perc', 0) > 0:
            fund_management_charge = current_corpus * (ulip_charges['fund_management_perc'] / 100 / 12)
            current_corpus -= fund_management_charge
            total_charges_deducted += fund_management_charge

        # Apply monthly mortality charge (simplified, per lakh of cover, per month)
        # Assuming a fixed sum assured for mortality calculation. This is a simplification.
        if ulip_charges and ulip_charges.get('mortality_per_lakh_pa', 0) > 0:
            annual_premium = sip_amount_monthly * 12
            sum_assured_for_mortality = max(10 * annual_premium, 1000000) # Example: 10x annual premium or 10 lakhs
            
            mortality_charge_monthly = (sum_assured_for_mortality / 100000) * (ulip_charges['mortality_per_lakh_pa'] / 12)
            current_corpus -= mortality_charge_monthly
            total_charges_deducted += mortality_charge_monthly

        # Apply monthly growth
        current_corpus *= (1 + monthly_return_rate)
        
        # Ensure corpus doesn't go negative due to charges
        current_corpus = max(0, current_corpus)

    nominal_fv = current_corpus
    
    # Calculate real future value
    real_fv = nominal_fv / ((1 + inflation_rate)**investment_years) if inflation_rate > 0 else nominal_fv

    return nominal_fv, real_fv, total_charges_deducted, total_invested_amount

def calculate_swp_remaining_corpus(initial_corpus, swp_amount_monthly, swp_period_years, growth_rate_annual, inflation_rate=0):
    """
    Calculates the remaining corpus after Systematic Withdrawal Plan (SWP), considering inflation.

    Args:
        initial_corpus (float): The initial corpus before SWP starts.
        swp_amount_monthly (float): Monthly SWP amount.
        swp_period_years (int): Duration of SWP in years.
        growth_rate_annual (float): Annual growth rate of the remaining corpus.
        inflation_rate (float): Annual inflation rate.

    Returns:
        tuple: (Remaining corpus nominal, Remaining corpus real, Total withdrawals nominal)
    """
    if initial_corpus <= 0 or swp_period_years <= 0:
        return initial_corpus, initial_corpus, 0.0

    corpus = initial_corpus
    monthly_growth_rate = (1 + growth_rate_annual)**(1/12) - 1
    monthly_inflation_rate = (1 + inflation_rate)**(1/12) - 1 if inflation_rate > 0 else 0
    total_swp_months = swp_period_years * 12
    total_withdrawals_nominal = 0

    for month in range(total_swp_months):
        # Apply growth
        corpus = corpus * (1 + monthly_growth_rate)
        
        # Withdraw SWP amount
        withdrawal_this_month = swp_amount_monthly
        corpus -= withdrawal_this_month
        total_withdrawals_nominal += withdrawal_this_month

        if corpus < 0: # If corpus runs out
            total_withdrawals_nominal += corpus # Adjust for over-withdrawal if corpus went negative
            corpus = 0
            break
    
    final_corpus_nominal = max(0, corpus)
    final_corpus_real = final_corpus_nominal / ((1 + inflation_rate)**swp_period_years) if inflation_rate > 0 else final_corpus_nominal
    
    return final_corpus_nominal, final_corpus_real, total_withdrawals_nominal

def get_historical_cagrs_and_metrics(historical_data, investment_period_years, start_year=None, end_year=None):
    """
    Calculates min, max, average CAGRs, volatility, and max drawdown for all continuous periods
    of a given length within the historical data, or for a specified period.

    Args:
        historical_data (pd.DataFrame): DataFrame with 'Year' and 'Value' columns.
        investment_period_years (int): The length of the investment period to consider.
        start_year (int, optional): The starting year for a specific period.
        end_year (int, optional): The ending year for a specific period.

    Returns:
        tuple: (min_cagr, avg_cagr, max_cagr, avg_volatility, max_drawdown) as percentages/values.
    """
    if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if investment_period_years <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Sort data by year to ensure correct windowing
    historical_data = historical_data.sort_values(by='Year').reset_index(drop=True)

    cagrs = []
    volatilities = []
    drawdowns = []

    if start_year is not None and end_year is not None:
        # Calculate for a specific period
        filtered_data = historical_data[(historical_data['Year'] >= start_year) & (historical_data['Year'] <= end_year)]
        if len(filtered_data) < investment_period_years + 1: # Need start + end + years in between
            st.warning(f"Selected period from {start_year} to {end_year} does not have enough data for a {investment_period_years}-year investment period.")
            return 0.0, 0.0, 0.0, 0.0, 0.0

        initial_value = filtered_data.iloc[0]['Value']
        final_value = filtered_data.iloc[investment_period_years]['Value'] # Value after 'investment_period_years'
        
        cagr = calculate_cagr(initial_value, final_value, investment_period_years)
        if cagr is not None:
            cagrs.append(cagr)

        # Calculate returns for volatility and drawdown
        # Ensure there's enough data for percentage change
        if len(filtered_data) > 1:
            returns = filtered_data['Value'].pct_change().dropna()
            if not returns.empty:
                volatilities.append(returns.std() * np.sqrt(12) * 100) # Annualized monthly volatility
                
                # Calculate drawdown for the selected period
                peak = filtered_data['Value'].expanding(min_periods=1).max()
                dd = (filtered_data['Value'] - peak) / peak
                drawdowns.append(dd.min() * 100) # Max drawdown as percentage
    else:
        # Iterate through all possible continuous periods
        for i in range(len(historical_data) - investment_period_years):
            start_row = historical_data.iloc[i]
            end_row = historical_data.iloc[i + investment_period_years]

            initial_value = start_row['Value']
            final_value = end_row['Value']

            cagr = calculate_cagr(initial_value, final_value, investment_period_years)
            if cagr is not None:
                cagrs.append(cagr)

            # Calculate returns for volatility and drawdown for this window
            window_data = historical_data.iloc[i : i + investment_period_years + 1] # +1 to get returns for the period
            if len(window_data) > 1:
                returns = window_data['Value'].pct_change().dropna()
                if not returns.empty:
                    volatilities.append(returns.std() * np.sqrt(12) * 100) # Annualized monthly volatility
                    
                    # Calculate drawdown for this specific window
                    peak = window_data['Value'].expanding(min_periods=1).max()
                    dd = (window_data['Value'] - peak) / peak
                    drawdowns.append(dd.min() * 100) # Max drawdown as percentage

    if not cagrs:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    min_cagr = min(cagrs)
    avg_cagr = np.mean(cagrs)
    max_cagr = max(cagrs)
    avg_volatility = np.mean(volatilities) if volatilities else 0.0
    max_overall_drawdown = min(drawdowns) if drawdowns else 0.0 # Max drawdown is the largest negative

    return min_cagr, avg_cagr, max_cagr, avg_volatility, max_overall_drawdown

# --- Data Generation for Historical Returns ---

@st.cache_data # Cache the data generation to avoid re-running on every interaction
def generate_nifty_data():
    """
    Generates synthetic annual Nifty 50 historical data from 2001 to 2023.
    Assumes average annual return of 12% with 15% standard deviation.
    """
    years = range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1)
    data = {'Year': [], 'Value': []}
    current_value = BASE_STARTING_VALUE
    np.random.seed(42) # for reproducibility

    for year in years:
        data['Year'].append(year)
        data['Value'].append(current_value)
        # Simulate annual return (normal distribution around 12% avg, 15% std dev)
        annual_return = np.random.normal(0.12, 0.15)
        current_value *= (1 + annual_return)
    return pd.DataFrame(data)

@st.cache_data # Cache the data generation
def generate_gold_data():
    """
    Generates synthetic annual Gold historical data from 2001 to 2023.
    Assumes average annual return of 9% with 8% standard deviation.
    """
    years = range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR + 1)
    data = {'Year': [], 'Value': []}
    current_value = BASE_STARTING_VALUE
    np.random.seed(43) # for reproducibility

    for year in years:
        data['Year'].append(year)
        data['Value'].append(current_value)
        # Simulate annual return (normal distribution around 9% avg, 8% std dev)
        annual_return = np.random.normal(0.09, 0.08)
        current_value *= (1 + annual_return)
    return pd.DataFrame(data)

# --- Tab 1: ULIP Investment Calculation ---

def ulip_tab():
    """
    Streamlit tab for ULIP investment CAGR calculation.
    """
    st.header("ULIP Investment Calculator")
    st.write("Calculate the CAGR for your ULIP investment based on your inputs, including charges and inflation.")

    # Input fields for ULIP
    col1, col2 = st.columns(2)
    with col1:
        sip_amount_monthly = st.number_input("Monthly ULIP Premium (SIP Amount)", min_value=1000.0, value=5000.0, step=500.0, key='ulip_sip',
                                             help="The regular monthly premium you pay for your ULIP.")
        investment_years = st.number_input("Period of Investment (Years)", min_value=1, value=10, step=1, key='ulip_inv_years',
                                           help="The total number of years you plan to pay premiums and invest.")
        assumed_ulip_growth_rate = st.slider("Assumed Annual ULIP Growth Rate (%)", min_value=0.0, max_value=20.0, value=8.0, step=0.5, key='ulip_growth',
                                             help="The expected annual growth rate of your ULIP fund before charges.") / 100
        inflation_rate = st.slider("Assumed Annual Inflation Rate (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='inflation_rate',
                                   help="The expected annual inflation rate, used to calculate real (inflation-adjusted) returns.") / 100
    with col2:
        swp_amount_monthly = st.number_input("Monthly SWP Amount (if applicable)", min_value=0.0, value=0.0, step=100.0, key='ulip_swp_amt',
                                             help="The monthly amount you plan to withdraw after the investment period (Systematic Withdrawal Plan).")
        swp_period_years = st.number_input("Period of SWP (Years)", min_value=0, value=0, step=1, key='ulip_swp_years',
                                           help="The duration in years for which you plan to make monthly withdrawals.")
        post_swp_growth_period_years = st.number_input("Post-SWP Growth Period (Years)", min_value=0, value=0, step=1, key='ulip_post_swp_growth',
                                                        help="Years the remaining corpus grows after SWP period ends but before final withdrawal. This is for scenarios where you withdraw for a period, and then let the rest of the corpus grow further.")

    st.subheader("ULIP Charges (Annual Percentage / Per Lakh)")
    col_charges1, col_charges2 = st.columns(2)
    with col_charges1:
        premium_allocation_charge = st.slider("Premium Allocation Charge (%)", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='ulip_charge_alloc',
                                              help="A percentage of your premium deducted upfront before investment.")
        policy_admin_monthly_perc = st.slider("Policy Administration Charge (Monthly % of Premium)", min_value=0.0, max_value=2.0, value=0.5, step=0.01, key='ulip_charge_admin',
                                              help="A fixed percentage of your monthly premium deducted for policy administration.")
    with col_charges2:
        fund_management_charge = st.slider("Fund Management Charge (Annual % of Fund Value)", min_value=0.0, max_value=3.0, value=1.5, step=0.01, key='ulip_charge_fm',
                                           help="An annual percentage deducted from your fund value for managing your investment.")
        mortality_per_lakh_pa = st.slider("Mortality Charge (Per Lakh of Sum Assured PA)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key='ulip_charge_mortality',
                                          help="An annual charge per lakh of sum assured, deducted for life cover. This is a simplified calculation.")

    ulip_charges = {
        'premium_allocation': premium_allocation_charge,
        'policy_admin_monthly_perc': policy_admin_monthly_perc,
        'fund_management_perc': fund_management_charge,
        'mortality_per_lakh_pa': mortality_per_lakh_pa
    }

    if st.button("Calculate ULIP Returns", key='calc_ulip_btn'):
        if investment_years <= 0:
            st.error("Period of Investment must be greater than 0.")
            st.session_state.ulip_result = {}
            return

        # Calculate corpus before SWP, considering charges
        corpus_before_swp_nominal, corpus_before_swp_real, total_charges_deducted, total_gross_invested_amount = calculate_sip_future_value(
            sip_amount_monthly, assumed_ulip_growth_rate, investment_years, inflation_rate, ulip_charges
        )
        st.session_state.ulip_result['total_invested_amount'] = total_gross_invested_amount
        st.session_state.ulip_result['corpus_before_swp_nominal'] = corpus_before_swp_nominal
        st.session_state.ulip_result['corpus_before_swp_real'] = corpus_before_swp_real
        st.session_state.ulip_result['total_charges_deducted'] = total_charges_deducted

        # Calculate corpus after SWP
        corpus_after_swp_nominal, corpus_after_swp_real, total_swp_withdrawals = calculate_swp_remaining_corpus(
            corpus_before_swp_nominal, swp_amount_monthly, swp_period_years, assumed_ulip_growth_rate, inflation_rate
        )
        st.session_state.ulip_result['corpus_after_swp_nominal'] = corpus_after_swp_nominal
        st.session_state.ulip_result['corpus_after_swp_real'] = corpus_after_swp_real
        st.session_state.ulip_result['total_swp_withdrawals'] = total_swp_withdrawals

        # If there's a post-SWP growth period, let the remaining corpus grow
        final_corpus_nominal = corpus_after_swp_nominal
        if post_swp_growth_period_years > 0:
            final_corpus_nominal *= (1 + assumed_ulip_growth_rate)**post_swp_growth_period_years
        
        final_corpus_real = final_corpus_nominal / ((1 + inflation_rate)**(investment_years + swp_period_years + post_swp_growth_period_years)) if inflation_rate > 0 else final_corpus_nominal

        st.session_state.ulip_result['final_corpus_nominal'] = final_corpus_nominal
        st.session_state.ulip_result['final_corpus_real'] = final_corpus_real

        # Total final value (remaining corpus + total withdrawals)
        total_final_value_nominal = final_corpus_nominal + total_swp_withdrawals
        # Inflation adjust the total final value based on the full period
        total_final_value_real = total_final_value_nominal / ((1 + inflation_rate)**(investment_years + swp_period_years + post_swp_growth_period_years)) if inflation_rate > 0 else total_final_value_nominal

        st.session_state.ulip_result['total_final_value_nominal'] = total_final_value_nominal
        st.session_state.ulip_result['total_final_value_real'] = total_final_value_real

        # Calculate effective CAGR based on total invested vs total final value
        total_effective_years = investment_years + swp_period_years + post_swp_growth_period_years
        ulip_cagr = calculate_cagr(total_gross_invested_amount, total_final_value_nominal, total_effective_years)
        st.session_state.ulip_result['cagr'] = ulip_cagr

        st.subheader("ULIP Investment Summary (Nominal Values)")
        st.write(f"**Total Gross Invested Amount:** ₹{total_gross_invested_amount:,.2f}")
        st.write(f"**Total Charges Deducted:** ₹{total_charges_deducted:,.2f}")
        st.write(f"**Corpus Before SWP:** ₹{corpus_before_swp_nominal:,.2f}")
        st.write(f"**Total SWP Withdrawals:** ₹{total_swp_withdrawals:,.2f}")
        st.write(f"**Remaining Corpus After SWP & Growth:** ₹{final_corpus_nominal:,.2f}")
        st.write(f"**Total Final Value (Corpus + Withdrawals):** ₹{total_final_value_nominal:,.2f}")
        st.markdown(f"**Calculated ULIP CAGR:** <span style='color:green; font-size: 20px;'>{ulip_cagr:.2f}%</span>", unsafe_allow_html=True)

        if inflation_rate > 0:
            st.subheader("ULIP Investment Summary (Inflation-Adjusted Real Values)")
            st.write(f"**Corpus Before SWP (Real):** ₹{corpus_before_swp_real:,.2f}")
            st.write(f"**Remaining Corpus After SWP & Growth (Real):** ₹{final_corpus_real:,.2f}")
            st.write(f"**Total Final Value (Real):** ₹{total_final_value_real:,.2f}")

        # Store results in session state for comparison tab
        st.session_state.ulip_result.update({
            'sip_amount_monthly': sip_amount_monthly,
            'investment_years': investment_years,
            'assumed_ulip_growth_rate': assumed_ulip_growth_rate,
            'swp_amount_monthly': swp_amount_monthly,
            'swp_period_years': swp_period_years,
            'post_swp_growth_period_years': post_swp_growth_period_years,
            'inflation_rate': inflation_rate,
            'ulip_charges': ulip_charges
        })

# --- Tab 2: Nifty 50 Historical Returns ---

def nifty_tab(nifty_data):
    """
    Streamlit tab for Nifty 50 historical returns analysis.
    """
    st.header("Nifty 50 Historical Returns")
    st.write(f"Analyze Nifty 50 returns from {HISTORICAL_START_YEAR} to {HISTORICAL_END_YEAR} based on synthetic historical data.")

    # Display historical data (optional, for transparency)
    st.subheader("Synthetic Nifty 50 Historical Data (Annual Values)")
    st.dataframe(nifty_data.set_index('Year').style.format({"Value": "₹{:,.2f}"}))

    st.subheader("Select Investment Period")
    col_nifty_period1, col_nifty_period2 = st.columns(2)
    with col_nifty_period1:
        nifty_start_year = st.select_slider(
            "Start Year for Nifty 50 Investment",
            options=list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR)),
            value=HISTORICAL_START_YEAR,
            key='nifty_start_year',
            help=f"Select the starting year for your Nifty 50 investment period (from {HISTORICAL_START_YEAR} to {HISTORICAL_END_YEAR-1})."
        )
    with col_nifty_period2:
        nifty_end_year = st.select_slider(
            "End Year for Nifty 50 Investment",
            options=list(range(HISTORICAL_START_YEAR + 1, HISTORICAL_END_YEAR + 1)),
            value=HISTORICAL_END_YEAR,
            key='nifty_end_year',
            help=f"Select the ending year for your Nifty 50 investment period (from {HISTORICAL_START_YEAR+1} to {HISTORICAL_END_YEAR})."
        )
    
    investment_period_nifty = nifty_end_year - nifty_start_year
    st.info(f"Selected Investment Period: **{investment_period_nifty} years** (from {nifty_start_year} to {nifty_end_year})")

    st.subheader("Lumpsum Investment Details (for Nifty 50)")
    nifty_lumpsum_amount = st.number_input("Lumpsum Investment Amount (Nifty 50)", min_value=1000.0, value=100000.0, step=1000.0, key='nifty_lumpsum_amt',
                                           help="The one-time lumpsum amount you wish to invest in Nifty 50.")
    nifty_swp_amount_monthly = st.number_input("Monthly SWP Amount from Nifty 50 Lumpsum (if applicable)", min_value=0.0, value=0.0, step=100.0, key='nifty_swp_amt',
                                               help="The monthly amount you plan to withdraw from your Nifty 50 lumpsum investment.")
    nifty_swp_period_years = st.number_input("Period of SWP from Nifty 50 Lumpsum (Years)", min_value=0, value=0, step=1, key='nifty_swp_years',
                                             help="The duration in years for which you plan to make monthly withdrawals from your Nifty 50 lumpsum.")

    if st.button("Calculate Nifty 50 Returns", key='calc_nifty_btn'):
        if investment_period_nifty <= 0:
            st.error("Selected investment period must be greater than 0.")
            st.session_state.nifty_results = {}
            return
        
        # Calculate for the specific selected period
        min_cagr, avg_cagr, max_cagr, avg_volatility, max_drawdown = get_historical_cagrs_and_metrics(
            nifty_data, investment_period_nifty, nifty_start_year, nifty_end_year
        )

        st.subheader(f"Nifty 50 Returns for {investment_period_nifty} Year Period ({nifty_start_year}-{nifty_end_year})")
        st.write(f"**Calculated Average CAGR:** <span style='color:blue; font-size: 20px;'>{avg_cagr:.2f}%</span>", unsafe_allow_html=True)
        st.write(f"**Historical Volatility (Annualized):** <span style='color:orange; font-size: 20px;'>{avg_volatility:.2f}%</span>", unsafe_allow_html=True)
        st.write(f"**Maximum Drawdown:** <span style='color:red; font-size: 20px;'>{max_drawdown:.2f}%</span>", unsafe_allow_html=True)
        
        if min_cagr != avg_cagr or max_cagr != avg_cagr: # Only show if there are multiple periods to average
             st.write(f"*(Min CAGR: {min_cagr:.2f}%, Max CAGR: {max_cagr:.2f}%)*")

        # Lumpsum calculation for Nifty 50
        st.subheader("Nifty 50 Lumpsum Investment Analysis")
        projected_final_value_lumpsum = 0
        corpus_after_swp_nifty_nominal = 0
        total_swp_withdrawals_nifty = 0

        if nifty_lumpsum_amount > 0:
            # Use the calculated average CAGR to project lumpsum growth
            projected_final_value_lumpsum = nifty_lumpsum_amount * (1 + avg_cagr / 100)**investment_period_nifty
            st.write(f"Projected Lumpsum Final Value (using Average CAGR): ₹{projected_final_value_lumpsum:,.2f}")

            # Calculate remaining corpus after SWP for lumpsum
            if nifty_swp_amount_monthly > 0 and nifty_swp_period_years > 0:
                # If SWP period is less than investment period, the remaining corpus grows for the difference
                effective_growth_years_after_swp = max(0, investment_period_nifty - nifty_swp_period_years)

                corpus_after_swp_nifty_nominal, _, total_swp_withdrawals_nifty = calculate_swp_remaining_corpus(
                    projected_final_value_lumpsum, nifty_swp_amount_monthly, nifty_swp_period_years, avg_cagr / 100
                )

                # If SWP period is less than investment period, the remaining corpus grows for the difference
                if effective_growth_years_after_swp > 0:
                    corpus_after_swp_nifty_nominal *= (1 + avg_cagr / 100)**effective_growth_years_after_swp

                total_final_value_nifty_lumpsum = corpus_after_swp_nifty_nominal + total_swp_withdrawals_nifty

                st.write(f"**Lumpsum Remaining After SWP (and subsequent growth):** ₹{corpus_after_swp_nifty_nominal:,.2f}")
                st.write(f"**Total SWP Withdrawals from Lumpsum:** ₹{total_swp_withdrawals_nifty:,.2f}")
                st.write(f"**Total Value (Lumpsum Final Value + SWP):** ₹{total_final_value_nifty_lumpsum:,.2f}")
            else:
                st.write("No SWP applied for Lumpsum investment.")
                total_final_value_nifty_lumpsum = projected_final_value_lumpsum

        # Year-by-year growth visualization
        st.subheader("Nifty 50 Year-by-Year Growth (Selected Period)")
        filtered_data_for_plot = nifty_data[(nifty_data['Year'] >= nifty_start_year) & (nifty_data['Year'] <= nifty_end_year)]
        if not filtered_data_for_plot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(filtered_data_for_plot['Year'], filtered_data_for_plot['Value'], marker='o', linestyle='-', color='purple')
            ax.set_title(f"Nifty 50 Value Growth ({nifty_start_year}-{nifty_end_year})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Index Value (Base 1000)")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Not enough data for year-by-year growth visualization in the selected period.")

        # Store results in session state for comparison tab
        st.session_state.nifty_results = {
            'min_cagr': min_cagr,
            'avg_cagr': avg_cagr,
            'max_cagr': max_cagr,
            'avg_volatility': avg_volatility,
            'max_drawdown': max_drawdown,
            'investment_period': investment_period_nifty,
            'lumpsum_amount': nifty_lumpsum_amount,
            'lumpsum_final_value': total_final_value_nifty_lumpsum, # This includes SWP if applicable
            'lumpsum_remaining_after_swp': corpus_after_swp_nifty_nominal,
            'total_swp_withdrawals': total_swp_withdrawals_nifty
        }

# --- Tab 3: Gold Historical Returns ---

def gold_tab(gold_data):
    """
    Streamlit tab for Gold historical returns analysis.
    """
    st.header("Gold Historical Returns")
    st.write(f"Analyze Gold returns from {HISTORICAL_START_YEAR} to {HISTORICAL_END_YEAR} based on synthetic historical data.")

    # Display historical data (optional)
    st.subheader("Synthetic Gold Historical Data (Annual Values)")
    st.dataframe(gold_data.set_index('Year').style.format({"Value": "₹{:,.2f}"}))

    st.subheader("Select Investment Period")
    col_gold_period1, col_gold_period2 = st.columns(2)
    with col_gold_period1:
        gold_start_year = st.select_slider(
            "Start Year for Gold Investment",
            options=list(range(HISTORICAL_START_YEAR, HISTORICAL_END_YEAR)),
            value=HISTORICAL_START_YEAR,
            key='gold_start_year',
            help=f"Select the starting year for your Gold investment period (from {HISTORICAL_START_YEAR} to {HISTORICAL_END_YEAR-1})."
        )
    with col_gold_period2:
        gold_end_year = st.select_slider(
            "End Year for Gold Investment",
            options=list(range(HISTORICAL_START_YEAR + 1, HISTORICAL_END_YEAR + 1)),
            value=HISTORICAL_END_YEAR,
            key='gold_end_year',
            help=f"Select the ending year for your Gold investment period (from {HISTORICAL_START_YEAR+1} to {HISTORICAL_END_YEAR})."
        )
    
    investment_period_gold = gold_end_year - gold_start_year
    st.info(f"Selected Investment Period: **{investment_period_gold} years** (from {gold_start_year} to {gold_end_year})")

    st.subheader("Lumpsum Investment Details (for Gold)")
    gold_lumpsum_amount = st.number_input("Lumpsum Investment Amount (Gold)", min_value=1000.0, value=100000.0, step=1000.0, key='gold_lumpsum_amt',
                                          help="The one-time lumpsum amount you wish to invest in Gold.")
    gold_swp_amount_monthly = st.number_input("Monthly SWP Amount from Gold Lumpsum (if applicable)", min_value=0.0, value=0.0, step=100.0, key='gold_swp_amt',
                                              help="The monthly amount you plan to withdraw from your Gold lumpsum investment.")
    gold_swp_period_years = st.number_input("Period of SWP from Gold Lumpsum (Years)", min_value=0, value=0, step=1, key='gold_swp_years',
                                            help="The duration in years for which you plan to make monthly withdrawals from your Gold lumpsum.")

    if st.button("Calculate Gold Returns", key='calc_gold_btn'):
        if investment_period_gold <= 0:
            st.error("Selected investment period must be greater than 0.")
            st.session_state.gold_results = {}
            return

        min_cagr, avg_cagr, max_cagr, avg_volatility, max_drawdown = get_historical_cagrs_and_metrics(
            gold_data, investment_period_gold, gold_start_year, gold_end_year
        )

        st.subheader(f"Gold Returns for {investment_period_gold} Year Period ({gold_start_year}-{gold_end_year})")
        st.write(f"**Calculated Average CAGR:** <span style='color:blue; font-size: 20px;'>{avg_cagr:.2f}%</span>", unsafe_allow_html=True)
        st.write(f"**Historical Volatility (Annualized):** <span style='color:orange; font-size: 20px;'>{avg_volatility:.2f}%</span>", unsafe_allow_html=True)
        st.write(f"**Maximum Drawdown:** <span style='color:red; font-size: 20px;'>{max_drawdown:.2f}%</span>", unsafe_allow_html=True)
        
        if min_cagr != avg_cagr or max_cagr != avg_cagr:
            st.write(f"*(Min CAGR: {min_cagr:.2f}%, Max CAGR: {max_cagr:.2f}%)*")

        # Lumpsum calculation for Gold
        st.subheader("Gold Lumpsum Investment Analysis")
        projected_final_value_lumpsum = 0
        corpus_after_swp_gold_nominal = 0
        total_swp_withdrawals_gold = 0

        if gold_lumpsum_amount > 0:
            projected_final_value_lumpsum = gold_lumpsum_amount * (1 + avg_cagr / 100)**investment_period_gold
            st.write(f"Projected Lumpsum Final Value (using Average CAGR): ₹{projected_final_value_lumpsum:,.2f}")

            if gold_swp_amount_monthly > 0 and gold_swp_period_years > 0:
                effective_growth_years_after_swp = max(0, investment_period_gold - gold_swp_period_years)

                corpus_after_swp_gold_nominal, _, total_swp_withdrawals_gold = calculate_swp_remaining_corpus(
                    projected_final_value_lumpsum, gold_swp_amount_monthly, gold_swp_period_years, avg_cagr / 100
                )

                if effective_growth_years_after_swp > 0:
                    corpus_after_swp_gold_nominal *= (1 + avg_cagr / 100)**effective_growth_years_after_swp

                total_final_value_gold_lumpsum = corpus_after_swp_gold_nominal + total_swp_withdrawals_gold

                st.write(f"**Lumpsum Remaining After SWP (and subsequent growth):** ₹{corpus_after_swp_gold_nominal:,.2f}")
                st.write(f"**Total SWP Withdrawals from Lumpsum:** ₹{total_swp_withdrawals_gold:,.2f}")
                st.write(f"**Total Value (Lumpsum Final Value + SWP):** ₹{total_final_value_gold_lumpsum:,.2f}")
            else:
                st.write("No SWP applied for Lumpsum investment.")
                total_final_value_gold_lumpsum = projected_final_value_lumpsum

        # Year-by-year growth visualization
        st.subheader("Gold Year-by-Year Growth (Selected Period)")
        filtered_data_for_plot = gold_data[(gold_data['Year'] >= gold_start_year) & (gold_data['Year'] <= gold_end_year)]
        if not filtered_data_for_plot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(filtered_data_for_plot['Year'], filtered_data_for_plot['Value'], marker='o', linestyle='-', color='gold')
            ax.set_title(f"Gold Value Growth ({gold_start_year}-{gold_end_year})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Index Value (Base 1000)")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("Not enough data for year-by-year growth visualization in the selected period.")

        # Store results in session state for comparison tab
        st.session_state.gold_results = {
            'min_cagr': min_cagr,
            'avg_cagr': avg_cagr,
            'max_cagr': max_cagr,
            'avg_volatility': avg_volatility,
            'max_drawdown': max_drawdown,
            'investment_period': investment_period_gold,
            'lumpsum_amount': gold_lumpsum_amount,
            'lumpsum_final_value': total_final_value_gold_lumpsum, # This includes SWP if applicable
            'lumpsum_remaining_after_swp': corpus_after_swp_gold_nominal,
            'total_swp_withdrawals': total_swp_withdrawals_gold
        }

# --- Tab 4: Risk Profile & Allocation ---

def risk_profile_tab():
    """
    Streamlit tab for Risk Profile Assessment and Portfolio Allocation.
    """
    st.header("Risk Profile & Portfolio Allocation")
    st.write("Answer a few questions to determine your investment risk tolerance and get a suggested asset allocation.")

    st.subheader("Risk Assessment Questionnaire")

    risk_score = 0

    q1 = st.radio(
        "1. What would you do if your investment portfolio dropped by 20% in a short period?",
        ("Sell everything to prevent further losses.",
         "Wait it out, hoping for a recovery.",
         "Invest more to average down and take advantage of lower prices."),
        key='q1_risk',
        help="Your reaction to a significant market downturn."
    )
    if q1 == "Sell everything to prevent further losses.": risk_score += 1
    elif q1 == "Wait it out, hoping for a recovery.": risk_score += 2
    elif q1 == "Invest more to average down and take advantage of lower prices.": risk_score += 3

    q2 = st.radio(
        "2. What is your primary investment goal?",
        ("Capital preservation (protecting my initial investment).",
         "Moderate growth with some risk.",
         "Aggressive growth with high risk."),
        key='q2_risk',
        help="Your main objective for investing your money."
    )
    if q2 == "Capital preservation (protecting my initial investment).": risk_score += 1
    elif q2 == "Moderate growth with some risk.": risk_score += 2
    elif q2 == "Aggressive growth with high risk.": risk_score += 3

    q3 = st.radio(
        "3. How long do you plan to invest?",
        ("Less than 3 years (Short-term).",
         "3-10 years (Medium-term).",
         "More than 10 years (Long-term)."),
        key='q3_risk',
        help="Your investment horizon."
    )
    if q3 == "Less than 3 years (Short-term).": risk_score += 1
    elif q3 == "3-10 years (Medium-term).": risk_score += 2
    elif q3 == "More than 10 years (Long-term).": risk_score += 3

    q4 = st.radio(
        "4. How comfortable are you with market volatility?",
        ("Very uncomfortable; I prefer stable returns.",
         "Somewhat comfortable; I can tolerate some ups and downs.",
         "Very comfortable; I see volatility as an opportunity."),
        key='q4_risk',
        help="Your emotional response to market fluctuations."
    )
    if q4 == "Very uncomfortable; I prefer stable returns.": risk_score += 1
    elif q4 == "Somewhat comfortable; I can tolerate some ups and downs.": risk_score += 2
    elif q4 == "Very comfortable; I see volatility as an opportunity.": risk_score += 3

    risk_profile = ""
    suggested_allocation = {}

    if risk_score <= 5:
        risk_profile = "Conservative"
        suggested_allocation = {"Equity": "20%", "Gold": "10%", "Debt/Fixed Income": "70%"}
    elif risk_score <= 8:
        risk_profile = "Moderate"
        suggested_allocation = {"Equity": "50%", "Gold": "15%", "Debt/Fixed Income": "35%"}
    else:
        risk_profile = "Aggressive"
        suggested_allocation = {"Equity": "70%", "Gold": "15%", "Debt/Fixed Income": "15%"}

    st.subheader("Your Risk Profile:")
    st.markdown(f"Based on your answers, your investment risk profile is: **{risk_profile}**")
    st.session_state.risk_profile_data['profile'] = risk_profile

    st.subheader("Suggested Portfolio Allocation:")
    df_allocation = pd.DataFrame(suggested_allocation.items(), columns=["Asset Class", "Allocation"])
    st.dataframe(df_allocation)
    st.session_state.risk_profile_data['allocation'] = suggested_allocation

    # Pie chart for allocation
    labels = list(suggested_allocation.keys())
    sizes = [float(val.strip('%')) for val in suggested_allocation.values()]
    colors = ['#4CAF50', '#FFD700', '#1E90FF'] # Green for Equity, Gold for Gold, Blue for Debt

    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
    ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    ax_pie.set_title("Suggested Portfolio Allocation")
    st.pyplot(fig_pie)

    st.subheader("Historical Risk Metrics (for reference)")
    st.write("These metrics indicate the historical riskiness of Nifty 50 and Gold based on the synthetic data.")

    nifty_results = st.session_state.get('nifty_results', {})
    gold_results = st.session_state.get('gold_results', {})

    if nifty_results and nifty_results.get('avg_volatility') is not None:
        st.write(f"**Nifty 50 (Avg. Volatility):** {nifty_results['avg_volatility']:.2f}%")
        st.write(f"**Nifty 50 (Max Drawdown):** {nifty_results['max_drawdown']:.2f}%")
    else:
        st.info("Nifty 50 historical risk metrics not calculated. Please visit 'Nifty 50 Historical' tab and calculate returns.")

    if gold_results and gold_results.get('avg_volatility') is not None:
        st.write(f"**Gold (Avg. Volatility):** {gold_results['avg_volatility']:.2f}%")
        st.write(f"**Gold (Max Drawdown):** {gold_results['max_drawdown']:.2f}%")
    else:
        st.info("Gold historical risk metrics not calculated. Please visit 'Gold Historical' tab and calculate returns.")


# --- Tab 5: Investment Comparison ---

def comparison_tab(ulip_result, nifty_results, gold_results):
    """
    Streamlit tab for comparing different investment products.
    """
    st.header("Investment Comparison")
    st.write("Compare the CAGR, risk, and overall returns of ULIP, Nifty 50, and Gold investments.")

    if not ulip_result and not nifty_results and not gold_results:
        st.info("Please calculate returns in the respective tabs first to see a comparison.")
        return

    st.subheader("Select CAGR Type for Comparison")
    col_nifty, col_gold = st.columns(2)

    with col_nifty:
        nifty_cagr_type = st.radio(
            "Nifty 50 CAGR Option:",
            ('Average CAGR', 'Minimum CAGR', 'Maximum CAGR'),
            key='comp_nifty_cagr_selection',
            help="Choose which Nifty 50 CAGR to use for comparison."
        )
    with col_gold:
        gold_cagr_type = st.radio(
            "Gold CAGR Option:",
            ('Average CAGR', 'Minimum CAGR', 'Maximum CAGR'),
            key='comp_gold_cagr_selection',
            help="Choose which Gold CAGR to use for comparison."
        )

    st.markdown("---")

    comparison_data = []

    # ULIP Data
    if ulip_result:
        ulip_cagr = ulip_result.get('cagr', 0.0)
        # For ULIP, we don't have historical volatility from data, so we'll use a proxy
        ulip_volatility = 5.0 # Placeholder for ULIP volatility (can be refined)
        ulip_max_drawdown = -10.0 # Placeholder for ULIP drawdown (can be refined)

        # Risk-adjusted return (simplified: CAGR / Volatility)
        ulip_risk_adjusted_return = ulip_cagr / ulip_volatility if ulip_volatility > 0 else 0

        comparison_data.append({
            "Investment Product": "ULIP",
            "CAGR (%)": ulip_cagr,
            "Volatility (%)": ulip_volatility,
            "Max Drawdown (%)": ulip_max_drawdown,
            "Risk-Adjusted Return (CAGR/Vol)": ulip_risk_adjusted_return,
            "Total Invested": ulip_result.get('total_invested_amount', 0.0),
            "Total Final Value": ulip_result.get('total_final_value_nominal', 0.0)
        })
    else:
        st.warning("ULIP results not available. Please calculate in 'ULIP Investment' tab.")

    # Nifty 50 Data
    if nifty_results:
        selected_nifty_cagr = 0.0
        if nifty_cagr_type == 'Average CAGR':
            selected_nifty_cagr = nifty_results.get('avg_cagr', 0.0)
        elif nifty_cagr_type == 'Minimum CAGR':
            selected_nifty_cagr = nifty_results.get('min_cagr', 0.0)
        elif nifty_cagr_type == 'Maximum CAGR':
            selected_nifty_cagr = nifty_results.get('max_cagr', 0.0)
        
        nifty_volatility = nifty_results.get('avg_volatility', 0.0)
        nifty_max_drawdown = nifty_results.get('max_drawdown', 0.0)
        nifty_risk_adjusted_return = selected_nifty_cagr / nifty_volatility if nifty_volatility > 0 else 0

        nifty_invested = nifty_results.get('lumpsum_amount', 0.0)
        nifty_final_value = nifty_results.get('lumpsum_final_value', 0.0) # This already includes SWP if applicable

        comparison_data.append({
            "Investment Product": f"Nifty 50 ({nifty_cagr_type})",
            "CAGR (%)": selected_nifty_cagr,
            "Volatility (%)": nifty_volatility,
            "Max Drawdown (%)": nifty_max_drawdown,
            "Risk-Adjusted Return (CAGR/Vol)": nifty_risk_adjusted_return,
            "Total Invested": nifty_invested,
            "Total Final Value": nifty_final_value
        })
    else:
        st.warning("Nifty 50 results not available. Please calculate in 'Nifty 50 Historical' tab.")

    # Gold Data
    if gold_results:
        selected_gold_cagr = 0.0
        if gold_cagr_type == 'Average CAGR':
            selected_gold_cagr = gold_results.get('avg_cagr', 0.0)
        elif gold_cagr_type == 'Minimum CAGR':
            selected_gold_cagr = gold_results.get('min_cagr', 0.0)
        elif gold_cagr_type == 'Maximum CAGR':
            selected_gold_cagr = gold_results.get('max_cagr', 0.0)
        
        gold_volatility = gold_results.get('avg_volatility', 0.0)
        gold_max_drawdown = gold_results.get('max_drawdown', 0.0)
        gold_risk_adjusted_return = selected_gold_cagr / gold_volatility if gold_volatility > 0 else 0

        gold_invested = gold_results.get('lumpsum_amount', 0.0)
        gold_final_value = gold_results.get('lumpsum_final_value', 0.0) # This already includes SWP if applicable

        comparison_data.append({
            "Investment Product": f"Gold ({gold_cagr_type})",
            "CAGR (%)": selected_gold_cagr,
            "Volatility (%)": gold_volatility,
            "Max Drawdown (%)": gold_max_drawdown,
            "Risk-Adjusted Return (CAGR/Vol)": gold_risk_adjusted_return,
            "Total Invested": gold_invested,
            "Total Final Value": gold_final_value
        })
    else:
        st.warning("Gold results not available. Please calculate in 'Gold Historical' tab.")

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison['Total Profit'] = df_comparison['Total Final Value'] - df_comparison['Total Invested']
        df_comparison = df_comparison.sort_values(by="CAGR (%)", ascending=False).reset_index(drop=True)

        st.subheader("Comparison Table")
        st.dataframe(df_comparison.style.format({
            "CAGR (%)": "{:.2f}%",
            "Volatility (%)": "{:.2f}%",
            "Max Drawdown (%)": "{:.2f}%",
            "Risk-Adjusted Return (CAGR/Vol)": "{:.2f}",
            "Total Invested": "₹{:,.2f}",
            "Total Final Value": "₹{:,.2f}",
            "Total Profit": "₹{:,.2f}"
        }))

        st.subheader("Analysis and Conclusion")
        if not df_comparison.empty:
            best_product_cagr = df_comparison.loc[df_comparison['CAGR (%)'].idxmax()]
            best_product_profit = df_comparison.loc[df_comparison['Total Profit'].idxmax()]
            best_product_risk_adjusted = df_comparison.loc[df_comparison['Risk-Adjusted Return (CAGR/Vol)'].idxmax()]


            st.markdown(f"Based on the selected parameters, the product with the **highest CAGR** is **{best_product_cagr['Investment Product']}** at **{best_product_cagr['CAGR (%)']:.2f}%**.")
            st.markdown(f"The product yielding the **maximum total profit** is **{best_product_profit['Investment Product']}** with a profit of **₹{best_product_profit['Total Profit']:.2f}**.")
            st.markdown(f"For a risk-adjusted perspective, **{best_product_risk_adjusted['Investment Product']}** shows the best return per unit of volatility, with a score of **{best_product_risk_adjusted['Risk-Adjusted Return (CAGR/Vol)']:.2f}**.")

            if len(df_comparison) > 1:
                # Calculate potential loss based on maximum profit product
                max_profit_value = best_product_profit['Total Profit']
                st.write("---")
                st.write("Potential Profit Loss Analysis (compared to the product with maximum total profit):")
                for i in range(len(df_comparison)):
                    other_product = df_comparison.iloc[i]
                    if other_product['Investment Product'] != best_product_profit['Investment Product']:
                        loss_amount = max_profit_value - other_product['Total Profit']
                        st.write(f"- By investing in **{other_product['Investment Product']}** instead of **{best_product_profit['Investment Product']}**, a user might potentially lose **₹{loss_amount:,.2f}** in profit.")
        else:
            st.info("No data available for comparison. Please ensure calculations are done in previous tabs.")
    else:
        st.info("No data to compare. Please ensure you have calculated results in the previous tabs.")


# --- Tab 6: MIS Report & Graphics ---

def mis_report_tab(ulip_result, nifty_results, gold_results, risk_profile_data):
    """
    Streamlit tab for Management Information System (MIS) Report and Graphics.
    """
    st.header("MIS Report & Graphics")
    st.write("Detailed report and graphical representation of your investment performance.")

    if not ulip_result and not nifty_results and not gold_results:
        st.info("Please calculate returns in the respective tabs first to generate the MIS report.")
        return

    st.subheader("Overall Investment Summary")
    # Collect all relevant data for the report
    report_data = []

    if ulip_result:
        # For MIS report, use nominal values
        ulip_cagr = ulip_result.get('cagr', 0.0)
        ulip_volatility = 5.0 # Placeholder
        ulip_max_drawdown = -10.0 # Placeholder
        ulip_risk_adjusted = ulip_cagr / ulip_volatility if ulip_volatility > 0 else 0

        report_data.append({
            "Product": "ULIP",
            "Investment Type": "SIP",
            "Total Invested": ulip_result.get('total_invested_amount', 0.0),
            "Total Final Value": ulip_result.get('total_final_value_nominal', 0.0),
            "Total Profit": ulip_result.get('total_final_value_nominal', 0.0) - ulip_result.get('total_invested_amount', 0.0),
            "CAGR (%)": ulip_cagr,
            "Volatility (%)": ulip_volatility,
            "Max Drawdown (%)": ulip_max_drawdown,
            "Risk-Adjusted Return (CAGR/Vol)": ulip_risk_adjusted,
            "Investment Period (Years)": ulip_result.get('investment_years', 0) + ulip_result.get('swp_period_years', 0) + ulip_result.get('post_swp_growth_period_years', 0)
        })
    if nifty_results:
        nifty_cagr = nifty_results.get('avg_cagr', 0.0)
        nifty_volatility = nifty_results.get('avg_volatility', 0.0)
        nifty_max_drawdown = nifty_results.get('max_drawdown', 0.0)
        nifty_risk_adjusted = nifty_cagr / nifty_volatility if nifty_volatility > 0 else 0

        report_data.append({
            "Product": "Nifty 50",
            "Investment Type": "Lumpsum",
            "Total Invested": nifty_results.get('lumpsum_amount', 0.0),
            "Total Final Value": nifty_results.get('lumpsum_final_value', 0.0),
            "Total Profit": nifty_results.get('lumpsum_final_value', 0.0) - nifty_results.get('lumpsum_amount', 0.0),
            "CAGR (%)": nifty_cagr,
            "Volatility (%)": nifty_volatility,
            "Max Drawdown (%)": nifty_max_drawdown,
            "Risk-Adjusted Return (CAGR/Vol)": nifty_risk_adjusted,
            "Investment Period (Years)": nifty_results.get('investment_period', 0)
        })
    if gold_results:
        gold_cagr = gold_results.get('avg_cagr', 0.0)
        gold_volatility = gold_results.get('avg_volatility', 0.0)
        gold_max_drawdown = gold_results.get('max_drawdown', 0.0)
        gold_risk_adjusted = gold_cagr / gold_volatility if gold_volatility > 0 else 0

        report_data.append({
            "Product": "Gold",
            "Investment Type": "Lumpsum",
            "Total Invested": gold_results.get('lumpsum_amount', 0.0),
            "Total Final Value": gold_results.get('lumpsum_final_value', 0.0),
            "Total Profit": gold_results.get('lumpsum_final_value', 0.0) - gold_results.get('lumpsum_amount', 0.0),
            "CAGR (%)": gold_cagr,
            "Volatility (%)": gold_volatility,
            "Max Drawdown (%)": gold_max_drawdown,
            "Risk-Adjusted Return (CAGR/Vol)": gold_risk_adjusted,
            "Investment Period (Years)": gold_results.get('investment_period', 0)
        })

    if report_data:
        df_report = pd.DataFrame(report_data)
        st.dataframe(df_report.style.format({
            "Total Invested": "₹{:,.2f}",
            "Total Final Value": "₹{:,.2f}",
            "Total Profit": "₹{:,.2f}",
            "CAGR (%)": "{:.2f}%",
            "Volatility (%)": "{:.2f}%",
            "Max Drawdown (%)": "{:.2f}%",
            "Risk-Adjusted Return (CAGR/Vol)": "{:.2f}"
        }))

        st.subheader("Graphical Representation of Returns")
        # Bar chart for invested vs. total return
        fig_returns, ax_returns = plt.subplots(figsize=(10, 6))
        
        products = df_report['Product']
        invested_amounts = df_report['Total Invested']
        final_values = df_report['Total Final Value']

        bar_width = 0.4
        index = np.arange(len(products))

        ax_returns.bar(index, invested_amounts, bar_width, label='Total Invested', color='orange', zorder=2)
        profit_amounts = final_values - invested_amounts
        ax_returns.bar(index, profit_amounts, bar_width, bottom=invested_amounts, label='Return Generated', color='navy', zorder=2)

        ax_returns.set_xlabel("Investment Product")
        ax_returns.set_ylabel("Amount (₹)")
        ax_returns.set_title("Investment vs. Total Return Generated")
        ax_returns.set_xticks(index)
        ax_returns.set_xticklabels(products)
        ax_returns.legend()
        ax_returns.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_returns)

        # Bar chart for Risk-Adjusted Returns
        st.subheader("Risk-Adjusted Returns Comparison")
        fig_risk_adj, ax_risk_adj = plt.subplots(figsize=(10, 6))
        ax_risk_adj.bar(df_report['Product'], df_report['Risk-Adjusted Return (CAGR/Vol)'], color=['#4CAF50', '#1E90FF', '#FFD700'])
        ax_risk_adj.set_xlabel("Investment Product")
        ax_risk_adj.set_ylabel("Risk-Adjusted Return (CAGR/Vol)")
        ax_risk_adj.set_title("Risk-Adjusted Returns (Higher is Better)")
        ax_risk_adj.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_risk_adj)

        # Suggested Portfolio Allocation Pie Chart (from Risk Profile tab)
        if risk_profile_data and risk_profile_data.get('allocation'):
            st.subheader("Your Suggested Portfolio Allocation")
            labels = list(risk_profile_data['allocation'].keys())
            sizes = [float(val.strip('%')) for val in risk_profile_data['allocation'].values()]
            colors = ['#4CAF50', '#FFD700', '#1E90FF']

            fig_pie_mis, ax_pie_mis = plt.subplots(figsize=(8, 8))
            ax_pie_mis.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
            ax_pie_mis.axis('equal')
            ax_pie_mis.set_title(f"Suggested Portfolio Allocation for {risk_profile_data.get('profile', 'Your')} Profile")
            st.pyplot(fig_pie_mis)


        st.subheader("Detailed Analysis")
        st.markdown("""
        This section provides a summary and analysis of the investment products based on the calculated returns.
        """)
        for index, row in df_report.iterrows():
            st.markdown(f"#### {row['Product']} ({row['Investment Type']})")
            st.markdown(f"- **Total Invested:** ₹{row['Total Invested']:,.2f}")
            st.markdown(f"- **Total Final Value:** ₹{row['Total Final Value']:,.2f}")
            st.markdown(f"- **Total Profit:** ₹{row['Total Profit']:,.2f}")
            st.markdown(f"- **CAGR:** {row['CAGR (%)']:.2f}%")
            st.markdown(f"- **Volatility:** {row['Volatility (%)']:.2f}%")
            st.markdown(f"- **Max Drawdown:** {row['Max Drawdown (%)']:.2f}%")
            st.markdown(f"- **Risk-Adjusted Return (CAGR/Vol):** {row['Risk-Adjusted Return (CAGR/Vol)']:.2f}")
            st.markdown(f"- **Investment Period:** {row['Investment Period (Years)']} years")
            st.markdown("---")

        st.markdown("### Key Insights:")
        if not df_report.empty:
            max_cagr_product = df_report.loc[df_report['CAGR (%)'].idxmax()]
            max_profit_product = df_report.loc[df_report['Total Profit'].idxmax()]
            best_risk_adjusted_product = df_report.loc[df_report['Risk-Adjusted Return (CAGR/Vol)'].idxmax()]

            st.markdown(f"- The investment product with the **highest CAGR** is **{max_cagr_product['Product']}** at **{max_cagr_product['CAGR (%)']:.2f}%**.")
            st.markdown(f"- The investment product yielding the **highest total profit** is **{max_profit_product['Product']}** with a profit of **₹{max_profit_product['Total Profit']:.2f}**.")
            st.markdown(f"- From a risk-adjusted perspective, **{best_risk_adjusted_product['Product']}** offers the best return per unit of risk, with a score of **{best_risk_adjusted_product['Risk-Adjusted Return (CAGR/Vol)']:.2f}**.")

            if len(df_report) > 1:
                # Calculate potential loss based on maximum profit product
                max_profit_value = max_profit_product['Total Profit']
                st.markdown("#### Potential Profit Loss by Choosing Sub-optimal Investments:")
                for index, row in df_report.iterrows():
                    if row['Product'] != max_profit_product['Product']:
                        profit_loss = max_profit_value - row['Total Profit']
                        st.markdown(f"- By investing in **{row['Product']}** instead of **{max_profit_product['Product']}**, a user might potentially lose **₹{profit_loss:,.2f}** in profit.")
        else:
            st.info("No data available for detailed analysis. Please ensure calculations are done in previous tabs.")

        # Download Report Button
        csv_report = df_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download MIS Report as CSV",
            data=csv_report,
            file_name="investment_mis_report.csv",
            mime="text/csv",
            key='download_mis_report'
        )

    else:
        st.info("No data available to generate the MIS report. Please ensure calculations are done in previous tabs.")

# --- Tab 7: AI Investment Assistant ---

async def ai_assistant_tab(ulip_result, nifty_results, gold_results, risk_profile_data):
    """
    Streamlit tab for AI Investment Assistant.
    """
    st.header("AI Investment Assistant")
    st.write("Ask the AI assistant questions about your investment analysis, get insights, or explanations.")

    user_query = st.text_area("Your Question:", key='ai_query_input', height=100,
                              help="Ask about your results, compare products, or understand financial terms. E.g., 'Explain my ULIP returns', 'Which investment is best for my risk profile?', 'What is maximum drawdown?'")

    if st.button("Get AI Insight", key='get_ai_insight_btn'):
        if not user_query:
            st.warning("Please enter a question for the AI assistant.")
            return

        # Prepare context for the AI
        context = "Here is the current investment analysis data:\n\n"
        
        if ulip_result:
            context += f"ULIP Results: Total Invested={ulip_result.get('total_invested_amount', 0):.2f}, Final Value={ulip_result.get('total_final_value_nominal', 0):.2f}, CAGR={ulip_result.get('cagr', 0):.2f}%, Charges Deducted={ulip_result.get('total_charges_deducted', 0):.2f}, Inflation Rate={ulip_result.get('inflation_rate', 0)*100:.2f}%. Investment Period={ulip_result.get('investment_years',0)} years.\n"
        if nifty_results:
            context += f"Nifty 50 Results: Lumpsum Invested={nifty_results.get('lumpsum_amount', 0):.2f}, Final Value={nifty_results.get('lumpsum_final_value', 0):.2f}, Avg CAGR={nifty_results.get('avg_cagr', 0):.2f}%, Volatility={nifty_results.get('avg_volatility', 0):.2f}%, Max Drawdown={nifty_results.get('max_drawdown', 0):.2f}%. Investment Period={nifty_results.get('investment_period',0)} years.\n"
        if gold_results:
            context += f"Gold Results: Lumpsum Invested={gold_results.get('lumpsum_amount', 0):.2f}, Final Value={gold_results.get('lumpsum_final_value', 0):.2f}, Avg CAGR={gold_results.get('avg_cagr', 0):.2f}%, Volatility={gold_results.get('avg_volatility', 0):.2f}%, Max Drawdown={gold_results.get('max_drawdown', 0):.2f}%. Investment Period={gold_results.get('investment_period',0)} years.\n"
        if risk_profile_data and risk_profile_data.get('profile'):
            context += f"Your Risk Profile: {risk_profile_data.get('profile')}. Suggested Allocation: {risk_profile_data.get('allocation')}.\n"

        if not ulip_result and not nifty_results and not gold_results and not risk_profile_data:
            context += "No investment calculations have been performed yet. Please calculate returns in other tabs first."

        full_prompt = f"""
        You are a helpful and knowledgeable AI financial assistant. Your goal is to analyze the provided investment data and user query to offer clear, concise, and actionable insights or explanations.
        
        Here is the current state of the user's investment analysis:
        {context}

        User's Question: {user_query}

        Please provide your response based on the available data. If data for a specific product is missing, mention that. Keep your response professional and easy to understand.
        """

        st.info("AI is thinking... Please wait.")
        ai_response_placeholder = st.empty() # Placeholder for the AI response

        try:
            # Gemini API call
            chatHistory = []
            chatHistory.append({ "role": "user", "parts": [{ "text": full_prompt }] })
            payload = { "contents": chatHistory }
            apiKey = GEMINI_API_KEY # Use the global API key variable
            apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"
            
            response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result.get('candidates') and len(result['candidates']) > 0 and \
               result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
               len(result['candidates'][0]['content']['parts']) > 0:
                ai_text = result['candidates'][0]['content']['parts'][0]['text']
                ai_response_placeholder.markdown(f"**AI Assistant's Response:**\n{ai_text}")
            else:
                ai_response_placeholder.error("AI could not generate a response. Please try again.")
                st.error(f"Unexpected API response structure: {result}")

        except requests.exceptions.RequestException as e:
            ai_response_placeholder.error(f"Error communicating with AI: {e}")
            st.error("Please ensure your API key is correctly configured or try again later.")
        except Exception as e:
            ai_response_placeholder.error(f"An unexpected error occurred: {e}")


# --- Main Application Logic ---

def main():
    """
    Main function to run the Streamlit application.
    Initializes data and sets up tabs.
    """
    st.sidebar.title("App Controls")
    if st.sidebar.button("Reset All Data", key='reset_all_data_btn', help="Clears all input fields and calculated results."):
        st.session_state.ulip_result = {}
        st.session_state.nifty_results = {}
        st.session_state.gold_results = {}
        st.session_state.risk_profile_data = {}
        st.session_state.nifty_historical_data = generate_nifty_data() # Regenerate historical data
        st.session_state.gold_historical_data = generate_gold_data() # Regenerate historical data
        st.experimental_rerun() # Rerun the app to clear inputs

    st.title("Enhanced Investment Analysis Dashboard")
    st.markdown("Welcome to the Investment Analysis Dashboard! Use the tabs below to calculate and compare returns for various investment products.")

    # Generate historical data once and store in session state
    if 'nifty_historical_data' not in st.session_state:
        st.session_state.nifty_historical_data = generate_nifty_data()
    if 'gold_historical_data' not in st.session_state:
        st.session_state.gold_historical_data = generate_gold_data()

    # Initialize session state for results if not already present
    # This ensures results persist across tab switches
    if 'ulip_result' not in st.session_state:
        st.session_state.ulip_result = {}
    if 'nifty_results' not in st.session_state:
        st.session_state.nifty_results = {}
    if 'gold_results' not in st.session_state:
        st.session_state.gold_results = {}
    if 'risk_profile_data' not in st.session_state:
        st.session_state.risk_profile_data = {}


    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ULIP Investment",
        "Nifty 50 Historical",
        "Gold Historical",
        "Risk Profile & Allocation",
        "Investment Comparison",
        "MIS Report & Graphics",
        "AI Investment Assistant"
    ])

    # Render content for each tab
    with tab1:
        ulip_tab()
    with tab2:
        nifty_tab(st.session_state.nifty_historical_data)
    with tab3:
        gold_tab(st.session_state.gold_historical_data)
    with tab4:
        risk_profile_tab()
    with tab5:
        comparison_tab(st.session_state.ulip_result, st.session_state.nifty_results, st.session_state.gold_results)
    with tab6:
        mis_report_tab(st.session_state.ulip_result, st.session_state.nifty_results, st.session_state.gold_results, st.session_state.risk_profile_data)
    with tab7:
        # Use st.expander to make the AI assistant section collapsible
        with st.expander("AI Assistant Help", expanded=True):
            st.info("The AI assistant can answer questions based on the data you've entered and the results calculated in the other tabs. Make sure to calculate results in the relevant tabs before asking the AI for insights.")
        ai_assistant_tab(st.session_state.ulip_result, st.session_state.nifty_results, st.session_state.gold_results, st.session_state.risk_profile_data)

# Run the application
if __name__ == "__main__":
    main()
