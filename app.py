import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import PyPDF2
import pytesseract
from PIL import Image
import io
import requests
import json
from google.cloud import vision
from google.oauth2 import service_account
import time
import hashlib

# Load environment variables
load_dotenv()

# Initialize Perplexity API configuration
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Initialize Google Vision client
try:
    credentials = service_account.Credentials.from_service_account_file('google_credentials.json')
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
    GOOGLE_VISION_ENABLED = True
except Exception as e:
    st.warning(f"Google Vision API not configured: {str(e)}")
    GOOGLE_VISION_ENABLED = False

# Set page config
st.set_page_config(
    page_title="Tax Regime Selector",
    page_icon="💰",
    layout="wide"
)

# Constants for tax calculation
OLD_REGIME_SLABS = [
    (250000, 0),
    (500000, 0.05),
    (1000000, 0.20),
    (float('inf'), 0.30)
]

NEW_REGIME_SLABS = [
    (400000, 0),
    (800000, 0.05),
    (1200000, 0.10),
    (1600000, 0.15),
    (2000000, 0.20),
    (2400000, 0.25),
    (float('inf'), 0.30)
]

def calculate_tax(income_details, deductions, regime="old"):
    """Calculate tax based on the given regime, income details and deductions."""
    # Calculate total income
    total_income = (
        income_details.get('basic_salary', 0) +
        income_details.get('hra', 0) +
        income_details.get('special_allowance', 0) +
        income_details.get('bonus', 0)
    )
    
    # Calculate total deductions
    total_deductions = (
        deductions.get('section_80c', 0) +
        deductions.get('section_80d', 0) +
        deductions.get('home_loan_interest', 0) +
        deductions.get('rent_paid', 0)
    )
    
    # Apply standard deduction
    if regime == "old":
        standard_deduction = 50000
    else:
        standard_deduction = 75000
    
    # Calculate taxable income
    if regime == "old":
        taxable_income = max(0, total_income - total_deductions - standard_deduction)
    else:
        taxable_income = max(0, total_income - standard_deduction)  # New regime doesn't allow most deductions
    
    # Calculate tax based on regime
    if regime == "old":
        if taxable_income <= 250000:
            tax = 0
        elif taxable_income <= 500000:
            tax = 0.05 * (taxable_income - 250000)
        elif taxable_income <= 1000000:
            tax = 12500 + 0.20 * (taxable_income - 500000)
        else:
            tax = 112500 + 0.30 * (taxable_income - 1000000)
    else:  # new regime
        if taxable_income <= 300000:
            tax = 0
        elif taxable_income <= 600000:
            tax = 0.05 * (taxable_income - 300000)
        elif taxable_income <= 900000:
            tax = 15000 + 0.10 * (taxable_income - 600000)
        elif taxable_income <= 1200000:
            tax = 45000 + 0.15 * (taxable_income - 900000)
        elif taxable_income <= 1500000:
            tax = 90000 + 0.20 * (taxable_income - 1200000)
        else:
            tax = 150000 + 0.30 * (taxable_income - 1500000)
    
    # Add 4% cess
    tax = tax * 1.04
    
    return tax

def get_perplexity_response(prompt):
    """Get response from Perplexity API using requests with rate limiting and caching"""
    # Check if API key exists
    if not PERPLEXITY_API_KEY:
        st.error("Perplexity API key not found. Please set the PERPLEXITY_API_KEY environment variable.")
        return "API key not configured. Please contact support."

    # Rate limiting check
    current_time = time.time()
    if 'last_api_call_time' in st.session_state:
        time_since_last_call = current_time - st.session_state.last_api_call_time
        if time_since_last_call < 60:  # 1 minute cooldown
            remaining_time = int(60 - time_since_last_call)
            st.warning(f"Please wait {remaining_time} seconds before requesting more advice.")
            return "Rate limit exceeded. Please try again later."

    # Create cache key from prompt
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    
    # Check cache
    if 'ai_advice_cache' not in st.session_state:
        st.session_state.ai_advice_cache = {}
    
    if cache_key in st.session_state.ai_advice_cache:
        cached_response = st.session_state.ai_advice_cache[cache_key]
        if time.time() - cached_response['timestamp'] < 3600:  # Cache valid for 1 hour
            return cached_response['content']
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "You are a tax advisor. Provide clear, concise advice about Indian tax regimes."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        
        # Update rate limiting timestamp
        st.session_state.last_api_call_time = current_time
        
        # Cache the response
        st.session_state.ai_advice_cache[cache_key] = {
            'content': content,
            'timestamp': current_time
        }
        
        # Clean old cache entries (older than 1 hour)
        current_time = time.time()
        st.session_state.ai_advice_cache = {
            k: v for k, v in st.session_state.ai_advice_cache.items()
            if current_time - v['timestamp'] < 3600
        }
        
        return content
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting AI advice: {str(e)}")
        return "Unable to get AI advice at this time. Please try again later."
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return "An unexpected error occurred. Please try again later."

def extract_text_with_google_vision(image_content):
    """Extract text from image using Google Vision API"""
    try:
        image = vision.Image(content=image_content)
        response = vision_client.document_text_detection(image=image)
        return response.full_text_annotation.text
    except Exception as e:
        st.error(f"Error with Google Vision API: {str(e)}")
        return None

def extract_field(text, keywords, default=0):
    """Extract numeric values from text using various patterns"""
    import re
    for kw in keywords:
        # Try different patterns to improve extraction accuracy
        patterns = [
            rf"{kw}[^\d]*(\d+[\d,]*\.?\d*)",         # Standard pattern
            rf"{kw}.*?(\d+[\d,]*\.?\d*)",            # Looser pattern
            rf"{kw}.*?(?:Rs\.?|₹)?\s*(\d+[\d,]*\.?\d*)", # With currency symbol
            rf"(?:Rs\.?|₹)?\s*(\d+[\d,]*\.?\d*).*?{kw}"  # Currency symbol before value
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Clean up the value (remove commas)
                value_str = match.group(1).replace(",", "")
                try:
                    return float(value_str)
                except ValueError:
                    continue
    return default

def process_document_text(text):
    """Process extracted text to find salary and tax information"""
    # Basic extraction
    data = {
        "basic_salary": extract_field(text, ["Basic Salary", "Basic Pay", "Basic"]),
        "hra": extract_field(text, ["HRA", "House Rent Allowance"]),
        "special_allowance": extract_field(text, ["Special Allowance", "Special"]),
        "bonus": extract_field(text, ["Bonus", "Performance Bonus", "Annual Bonus"]),
        "section_80c": extract_field(text, ["Section 80C", "80C"]),
        "section_80d": extract_field(text, ["Section 80D", "80D", "Health Insurance"]),
        "home_loan_interest": extract_field(text, ["Home Loan Interest", "Interest on Housing Loan"]),
        "rent_paid": extract_field(text, ["Rent Paid", "Rent"]),
        # Additional fields
        "gross_salary": extract_field(text, ["Gross Salary", "Gross"]),
        "net_salary": extract_field(text, ["Net Salary", "Take Home", "Net Pay"]),
        "tax_deducted": extract_field(text, ["TDS", "Tax Deducted", "Income Tax"])
    }
    
    # Try to find total income if individual components are missing
    if data["gross_salary"] > 0 and data["basic_salary"] == 0:
        # Estimate basic salary as a percentage of gross (typically 40-60%)
        data["basic_salary"] = data["gross_salary"] * 0.5
    
    # Try to find HRA if missing
    if data["hra"] == 0 and data["basic_salary"] > 0:
        # Estimate HRA as a percentage of basic (typically 40-50%)
        data["hra"] = data["basic_salary"] * 0.4
    
    # Try to find special allowance if missing
    if data["special_allowance"] == 0 and data["gross_salary"] > 0 and data["basic_salary"] > 0 and data["hra"] > 0:
        # Estimate as remaining amount after basic and HRA
        data["special_allowance"] = max(0, data["gross_salary"] - data["basic_salary"] - data["hra"])
    
    return data

def calculate_extraction_confidence(extracted_data):
    """Calculate confidence score for extracted data"""
    # Count how many fields have non-zero values
    non_zero_fields = sum(1 for v in extracted_data.values() if v > 0)
    total_fields = len(extracted_data)
    
    # Calculate confidence score
    confidence_score = non_zero_fields / total_fields if total_fields > 0 else 0
    
    return confidence_score

def main():
    st.title("💰 Tax Regime Selector")
    st.markdown("""
    This application helps you compare tax liability under the old and new tax regimes
    to make an informed decision about which regime to choose.
    """)

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "Upload Form 16", "Upload Salary Slip"]
    )

    # Collect income details and deductions for all input methods
    income_details = {}
    deductions = {}
    selected_regime = ""

    if input_method == "Manual Entry":
        with st.form("tax_calculator"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Income Details")
                # Add input type selection
                income_input_type = st.radio(
                    "Select Income Input Type",
                    ["Monthly", "Yearly"],
                    horizontal=True
                )
                
                # Basic salary input with conversion
                basic_salary = st.number_input(
                    f"Basic Salary ({income_input_type})", 
                    min_value=0.0, 
                    step=10000.0
                )
                if income_input_type == "Monthly":
                    basic_salary = basic_salary * 12
                    st.info(f"Yearly Basic Salary: ₹{basic_salary:,.2f}")
                
                # HRA input with conversion
                hra = st.number_input(
                    f"HRA ({income_input_type})", 
                    min_value=0.0, 
                    step=10000.0
                )
                if income_input_type == "Monthly":
                    hra = hra * 12
                    st.info(f"Yearly HRA: ₹{hra:,.2f}")
                
                # Special allowance input with conversion
                special_allowance = st.number_input(
                    f"Special Allowance ({income_input_type})", 
                    min_value=0.0, 
                    step=10000.0
                )
                if income_input_type == "Monthly":
                    special_allowance = special_allowance * 12
                    st.info(f"Yearly Special Allowance: ₹{special_allowance:,.2f}")
                
                # Bonus input (typically yearly)
                bonus = st.number_input(
                    "Bonus (Yearly)", 
                    min_value=0.0, 
                    step=10000.0
                )
                
            with col2:
                st.subheader("Deductions")
                # Section 80C (yearly limit)
                section_80c = st.number_input(
                    "Section 80C Investments (Yearly)", 
                    min_value=0.0, 
                    max_value=150000.0, 
                    step=1000.0
                )
                
                # Section 80D (yearly)
                section_80d = st.number_input(
                    "Section 80D (Health Insurance) (Yearly)", 
                    min_value=0.0, 
                    step=1000.0
                )
                
                # Home loan interest (yearly)
                home_loan_interest = st.number_input(
                    "Home Loan Interest (Yearly)", 
                    min_value=0.0, 
                    max_value=200000.0, 
                    step=1000.0
                )
                
                # Rent paid (yearly)
                rent_paid = st.number_input(
                    "Annual Rent Paid", 
                    min_value=0.0, 
                    step=10000.0
                )

            submitted = st.form_submit_button("Calculate Tax")

            if submitted:
                # Collect income details
                income_details = {
                    'basic_salary': basic_salary,
                    'hra': hra,
                    'special_allowance': special_allowance,
                    'bonus': bonus,
                    'total_income': basic_salary + hra + special_allowance + bonus
                }
                
                # Collect deduction details
                deductions = {
                    'section_80c': section_80c,
                    'section_80d': section_80d,
                    'home_loan_interest': home_loan_interest,
                    'rent_paid': rent_paid,
                    'total_deductions': section_80c + section_80d + home_loan_interest
                }
                
                # Calculate tax for both regimes
                old_regime_tax = calculate_tax(income_details, deductions, "old")
                new_regime_tax = calculate_tax(income_details, deductions, "new")

                # Display results
                st.header("Tax Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Old Regime")
                    st.write(f"Total Income: ₹{income_details['total_income']:,.2f}")
                    st.write(f"Total Deductions: ₹{deductions['total_deductions']:,.2f}")
                    st.write(f"Standard Deduction: ₹50,000")
                    st.write(f"Taxable Income: ₹{income_details['total_income'] - deductions['total_deductions'] - 50000:,.2f}")
                    st.write(f"Total Tax: ₹{old_regime_tax:,.2f}")
                    
                    # Show tax calculation details
                    with st.expander("Show Tax Calculation Details (Old Regime)"):
                        st.write("Tax Slabs:")
                        st.write("Up to ₹2.5L: 0%")
                        st.write("₹2.5L–₹5L: 5%")
                        st.write("₹5L–₹10L: 20%")
                        st.write("Above ₹10L: 30%")
                        
                        # Show step-by-step calculation
                        taxable_income = max(0, income_details['total_income'] - deductions['total_deductions'] - 50000)
                        st.write(f"Taxable Income: ₹{taxable_income:,.2f}")
                        
                        if taxable_income <= 0:
                            st.write("Taxable income is zero or negative. No tax applicable.")
                        else:
                            if taxable_income <= 250000:
                                tax = 0
                                st.write("Tax: ₹0 (Income below ₹2.5L)")
                            elif taxable_income <= 500000:
                                tax = 0.05 * (taxable_income - 250000)
                                st.write(f"Tax: 5% of (₹{taxable_income:,.2f} - ₹250,000) = ₹{tax:,.2f}")
                            elif taxable_income <= 1000000:
                                tax = 12500 + 0.20 * (taxable_income - 500000)
                                st.write(f"Tax: ₹12,500 + 20% of (₹{taxable_income:,.2f} - ₹500,000) = ₹{tax:,.2f}")
                            else:
                                tax = 112500 + 0.30 * (taxable_income - 1000000)
                                st.write(f"Tax: ₹1,12,500 + 30% of (₹{taxable_income:,.2f} - ₹10,00,000) = ₹{tax:,.2f}")
                            
                            cess = tax * 0.04
                            total_tax = tax + cess
                            st.write(f"Base Tax: ₹{tax:,.2f}")
                            st.write(f"Health & Education Cess (4%): ₹{cess:,.2f}")
                            st.write(f"Total Tax: ₹{total_tax:,.2f}")

                with col2:
                    st.subheader("New Regime")
                    st.write(f"Total Income: ₹{income_details['total_income']:,.2f}")
                    st.write(f"Standard Deduction: ₹75,000")
                    st.write(f"Taxable Income: ₹{income_details['total_income'] - 75000:,.2f}")
                    st.write(f"Total Tax: ₹{new_regime_tax:,.2f}")
                    
                    # Show tax calculation details
                    with st.expander("Show Tax Calculation Details (New Regime)"):
                        st.write("Tax Slabs:")
                        st.write("Up to ₹4L: 0%")
                        st.write("₹4L–₹8L: 5%")
                        st.write("₹8L–₹12L: 10%")
                        st.write("₹12L–₹16L: 15%")
                        st.write("₹16L–₹20L: 20%")
                        st.write("₹20L–₹24L: 25%")
                        st.write("Above ₹24L: 30%")
                        
                        # Show step-by-step calculation
                        taxable_income = max(0, income_details['total_income'] - 75000)
                        st.write(f"Taxable Income: ₹{taxable_income:,.2f}")
                        
                        if taxable_income <= 0:
                            st.write("Taxable income is zero or negative. No tax applicable.")
                        else:
                            if taxable_income <= 400000:
                                tax = 0
                                st.write("Tax: ₹0 (Income below ₹4L)")
                            elif taxable_income <= 800000:
                                tax = 0.05 * (taxable_income - 400000)
                                st.write(f"Tax: 5% of (₹{taxable_income:,.2f} - ₹400,000) = ₹{tax:,.2f}")
                            elif taxable_income <= 1200000:
                                tax = 20000 + 0.10 * (taxable_income - 800000)
                                st.write(f"Tax: ₹20,000 + 10% of (₹{taxable_income:,.2f} - ₹800,000) = ₹{tax:,.2f}")
                            elif taxable_income <= 1600000:
                                tax = 60000 + 0.15 * (taxable_income - 1200000)
                                st.write(f"Tax: ₹60,000 + 15% of (₹{taxable_income:,.2f} - ₹12,00,000) = ₹{tax:,.2f}")
                            elif taxable_income <= 2000000:
                                tax = 120000 + 0.20 * (taxable_income - 1600000)
                                st.write(f"Tax: ₹1,20,000 + 20% of (₹{taxable_income:,.2f} - ₹16,00,000) = ₹{tax:,.2f}")
                            elif taxable_income <= 2400000:
                                tax = 200000 + 0.25 * (taxable_income - 2000000)
                                st.write(f"Tax: ₹2,00,000 + 25% of (₹{taxable_income:,.2f} - ₹20,00,000) = ₹{tax:,.2f}")
                            else:
                                tax = 300000 + 0.30 * (taxable_income - 2400000)
                                st.write(f"Tax: ₹3,00,000 + 30% of (₹{taxable_income:,.2f} - ₹24,00,000) = ₹{tax:,.2f}")
                            
                            cess = tax * 0.04
                            total_tax = tax + cess
                            st.write(f"Base Tax: ₹{tax:,.2f}")
                            st.write(f"Health & Education Cess (4%): ₹{cess:,.2f}")
                            st.write(f"Total Tax: ₹{total_tax:,.2f}")

                # Recommendation
                st.header("Recommendation")
                if old_regime_tax < new_regime_tax:
                    selected_regime = "Old Regime"
                    st.success(f"The Old Regime is better for you! You'll save ₹{new_regime_tax - old_regime_tax:,.2f}")
                else:
                    selected_regime = "New Regime"
                    st.success(f"The New Regime is better for you! You'll save ₹{old_regime_tax - new_regime_tax:,.2f}")
                
                # Add AI tax advice button outside the form
                if 'income_details' in locals() and 'deductions' in locals():
                    if st.button("Get AI Tax Advice"):
                        with st.spinner("Getting personalized tax advice..."):
                            # Format the data for the prompt
                            income_breakdown = f"""
                            Income Details:
                            - Basic Salary: ₹{income_details['basic_salary']:,.2f}
                            - HRA: ₹{income_details['hra']:,.2f}
                            - Special Allowance: ₹{income_details['special_allowance']:,.2f}
                            - Bonus: ₹{income_details['bonus']:,.2f}
                            - Total Income: ₹{sum(income_details.values()):,.2f}
                            
                            Deductions:
                            - Section 80C: ₹{deductions['section_80c']:,.2f}
                            - Section 80D: ₹{deductions['section_80d']:,.2f}
                            - Home Loan Interest: ₹{deductions['home_loan_interest']:,.2f}
                            - Rent Paid: ₹{deductions['rent_paid']:,.2f}
                            - Total Deductions: ₹{sum(deductions.values()):,.2f}
                            
                            Tax Amounts:
                            - Old Regime Tax: ₹{old_regime_tax:,.2f}
                            - New Regime Tax: ₹{new_regime_tax:,.2f}
                            """
                            
                            advice = get_perplexity_response(f"""Based on the following tax information, provide specific advice on tax optimization:
                            {income_breakdown}
                            
                            Please provide:
                            1. Analysis of current tax situation
                            2. Specific recommendations for tax optimization
                            3. Potential savings opportunities
                            4. Any relevant tax planning strategies
                            
                            Format the response in a clear, structured manner.""")
                            
                            st.subheader("AI Tax Advisor Recommendations")
                            st.write(advice)
                else:
                    st.info("Please calculate your tax first to get personalized AI tax advice.")

    elif input_method == "Upload Form 16":
        st.subheader("Upload Form 16 (PDF)")
        uploaded_file = st.file_uploader("Upload your Form 16 PDF", type=["pdf"])
        
        # Add guidance for better results
        with st.expander("Tips for better extraction"):
            st.markdown("""
            **For best results:**
            - Ensure the Form 16 is a text-based PDF (not scanned)
            - Make sure all pages are included
            - The document should be clear and readable
            - If extraction is poor, try uploading a different format
            """)
        
        if uploaded_file is not None:
            # Show file details
            file_details = {
                "Filename": uploaded_file.name,
                "File type": uploaded_file.type,
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("File details:", file_details)
            
            # Process the file
            with st.spinner("Processing Form 16..."):
                text = ""
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                st.success(f"Form 16 processed successfully. Extracted {len(text)} characters.")
            
            # Display extracted text in an expander for debugging
            with st.expander("View Extracted Text"):
                st.text(text)
            
            # Process the extracted text
            extracted_data = process_document_text(text)
            
            # Show extraction confidence
            confidence_score = calculate_extraction_confidence(extracted_data)
            if confidence_score < 0.3:
                st.error("Low extraction confidence. Please check the extracted text and manually enter values.")
            elif confidence_score < 0.6:
                st.warning("Medium extraction confidence. Please review the extracted values.")
            else:
                st.success(f"High extraction confidence ({confidence_score:.0%}). Please review the extracted values.")
            
            st.info("Please review and edit the extracted values as needed.")
            with st.form("form16_confirm"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Income Details")
                    basic_salary = st.number_input("Basic Salary (Annual)", value=float(extracted_data["basic_salary"]), min_value=0.0, step=10000.0)
                    hra = st.number_input("HRA (Annual)", value=float(extracted_data["hra"]), min_value=0.0, step=10000.0)
                    special_allowance = st.number_input("Special Allowance (Annual)", value=float(extracted_data["special_allowance"]), min_value=0.0, step=10000.0)
                    bonus = st.number_input("Bonus (Annual)", value=float(extracted_data["bonus"]), min_value=0.0, step=10000.0)
                with col2:
                    st.subheader("Deductions")
                    section_80c = st.number_input("Section 80C Investments", value=float(extracted_data["section_80c"]), min_value=0.0, max_value=150000.0, step=1000.0)
                    section_80d = st.number_input("Section 80D (Health Insurance)", value=float(extracted_data["section_80d"]), min_value=0.0, step=1000.0)
                    home_loan_interest = st.number_input("Home Loan Interest", value=float(extracted_data["home_loan_interest"]), min_value=0.0, max_value=200000.0, step=1000.0)
                    rent_paid = st.number_input("Annual Rent Paid", value=float(extracted_data["rent_paid"]), min_value=0.0, step=10000.0)
                submitted = st.form_submit_button("Calculate Tax")
                if submitted:
                    # Convert monthly values to yearly if needed
                    if slip_type == "Monthly":
                        basic_salary = basic_salary * 12
                        hra = hra * 12
                        special_allowance = special_allowance * 12
                        # Bonus is typically yearly, so no conversion needed
                        st.info("Monthly values have been converted to yearly for tax calculation")
                    
                    # Prepare data for tax calculation
                    income_details = {
                        'basic_salary': basic_salary,
                        'hra': hra,
                        'special_allowance': special_allowance,
                        'bonus': bonus,
                        'total_income': basic_salary + hra + special_allowance + bonus
                    }
                    
                    deductions = {
                        'section_80c': section_80c,
                        'section_80d': section_80d,
                        'home_loan_interest': home_loan_interest,
                        'rent_paid': rent_paid,
                        'total_deductions': section_80c + section_80d + home_loan_interest
                    }
                    
                    # Calculate tax for both regimes
                    old_regime_tax = calculate_tax(income_details, deductions, "old")
                    new_regime_tax = calculate_tax(income_details, deductions, "new")
                    st.header("Tax Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Old Regime")
                        st.write(f"Total Income: ₹{income_details['total_income']:,.2f}")
                        st.write(f"Total Deductions: ₹{deductions['total_deductions']:,.2f}")
                        st.write(f"Standard Deduction: ₹50,000")
                        st.write(f"Taxable Income: ₹{income_details['total_income'] - deductions['total_deductions'] - 50000:,.2f}")
                        st.write(f"Total Tax: ₹{old_regime_tax:,.2f}")
                        
                        # Show tax calculation details
                        with st.expander("Show Tax Calculation Details (Old Regime)"):
                            st.write("Tax Slabs:")
                            st.write("Up to ₹2.5L: 0%")
                            st.write("₹2.5L–₹5L: 5%")
                            st.write("₹5L–₹10L: 20%")
                            st.write("Above ₹10L: 30%")
                            
                            # Show step-by-step calculation
                            taxable_income = max(0, income_details['total_income'] - deductions['total_deductions'] - 50000)
                            st.write(f"Taxable Income: ₹{taxable_income:,.2f}")
                            
                            if taxable_income <= 0:
                                st.write("Taxable income is zero or negative. No tax applicable.")
                            else:
                                if taxable_income <= 250000:
                                    tax = 0
                                    st.write("Tax: ₹0 (Income below ₹2.5L)")
                                elif taxable_income <= 500000:
                                    tax = 0.05 * (taxable_income - 250000)
                                    st.write(f"Tax: 5% of (₹{taxable_income:,.2f} - ₹250,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 1000000:
                                    tax = 12500 + 0.20 * (taxable_income - 500000)
                                    st.write(f"Tax: ₹12,500 + 20% of (₹{taxable_income:,.2f} - ₹500,000) = ₹{tax:,.2f}")
                                else:
                                    tax = 112500 + 0.30 * (taxable_income - 1000000)
                                    st.write(f"Tax: ₹1,12,500 + 30% of (₹{taxable_income:,.2f} - ₹10,00,000) = ₹{tax:,.2f}")
                                
                                cess = tax * 0.04
                                total_tax = tax + cess
                                st.write(f"Base Tax: ₹{tax:,.2f}")
                                st.write(f"Health & Education Cess (4%): ₹{cess:,.2f}")
                                st.write(f"Total Tax: ₹{total_tax:,.2f}")

                    with col2:
                        st.subheader("New Regime")
                        st.write(f"Total Income: ₹{income_details['total_income']:,.2f}")
                        st.write(f"Standard Deduction: ₹75,000")
                        st.write(f"Taxable Income: ₹{income_details['total_income'] - 75000:,.2f}")
                        st.write(f"Total Tax: ₹{new_regime_tax:,.2f}")
                        
                        # Show tax calculation details
                        with st.expander("Show Tax Calculation Details (New Regime)"):
                            st.write("Tax Slabs:")
                            st.write("Up to ₹4L: 0%")
                            st.write("₹4L–₹8L: 5%")
                            st.write("₹8L–₹12L: 10%")
                            st.write("₹12L–₹16L: 15%")
                            st.write("₹16L–₹20L: 20%")
                            st.write("₹20L–₹24L: 25%")
                            st.write("Above ₹24L: 30%")
                            
                            # Show step-by-step calculation
                            taxable_income = max(0, income_details['total_income'] - 75000)
                            st.write(f"Taxable Income: ₹{taxable_income:,.2f}")
                            
                            if taxable_income <= 0:
                                st.write("Taxable income is zero or negative. No tax applicable.")
                            else:
                                if taxable_income <= 400000:
                                    tax = 0
                                    st.write("Tax: ₹0 (Income below ₹4L)")
                                elif taxable_income <= 800000:
                                    tax = 0.05 * (taxable_income - 400000)
                                    st.write(f"Tax: 5% of (₹{taxable_income:,.2f} - ₹400,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 1200000:
                                    tax = 20000 + 0.10 * (taxable_income - 800000)
                                    st.write(f"Tax: ₹20,000 + 10% of (₹{taxable_income:,.2f} - ₹800,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 1600000:
                                    tax = 60000 + 0.15 * (taxable_income - 1200000)
                                    st.write(f"Tax: ₹60,000 + 15% of (₹{taxable_income:,.2f} - ₹12,00,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 2000000:
                                    tax = 120000 + 0.20 * (taxable_income - 1600000)
                                    st.write(f"Tax: ₹1,20,000 + 20% of (₹{taxable_income:,.2f} - ₹16,00,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 2400000:
                                    tax = 200000 + 0.25 * (taxable_income - 2000000)
                                    st.write(f"Tax: ₹2,00,000 + 25% of (₹{taxable_income:,.2f} - ₹20,00,000) = ₹{tax:,.2f}")
                                else:
                                    tax = 300000 + 0.30 * (taxable_income - 2400000)
                                    st.write(f"Tax: ₹3,00,000 + 30% of (₹{taxable_income:,.2f} - ₹24,00,000) = ₹{tax:,.2f}")
                                
                                cess = tax * 0.04
                                total_tax = tax + cess
                                st.write(f"Base Tax: ₹{tax:,.2f}")
                                st.write(f"Health & Education Cess (4%): ₹{cess:,.2f}")
                                st.write(f"Total Tax: ₹{total_tax:,.2f}")
                        st.header("Recommendation")
                        if old_regime_tax < new_regime_tax:
                            selected_regime = "Old Regime"
                            st.success(f"The Old Regime is better for you! You'll save ₹{new_regime_tax - old_regime_tax:,.2f}")
                        else:
                            selected_regime = "New Regime"
                            st.success(f"The New Regime is better for you! You'll save ₹{old_regime_tax - new_regime_tax:,.2f}")
                        
                        # Add AI tax advice button outside the form
                        if 'income_details' in locals() and 'deductions' in locals():
                            if st.button("Get AI Tax Advice"):
                                with st.spinner("Getting personalized tax advice..."):
                                    # Format the data for the prompt
                                    income_breakdown = f"""
                                    Income Details:
                                    - Basic Salary: ₹{income_details['basic_salary']:,.2f}
                                    - HRA: ₹{income_details['hra']:,.2f}
                                    - Special Allowance: ₹{income_details['special_allowance']:,.2f}
                                    - Bonus: ₹{income_details['bonus']:,.2f}
                                    - Total Income: ₹{sum(income_details.values()):,.2f}
                                    
                                    Deductions:
                                    - Section 80C: ₹{deductions['section_80c']:,.2f}
                                    - Section 80D: ₹{deductions['section_80d']:,.2f}
                                    - Home Loan Interest: ₹{deductions['home_loan_interest']:,.2f}
                                    - Rent Paid: ₹{deductions['rent_paid']:,.2f}
                                    - Total Deductions: ₹{sum(deductions.values()):,.2f}
                                    
                                    Tax Amounts:
                                    - Old Regime Tax: ₹{old_regime_tax:,.2f}
                                    - New Regime Tax: ₹{new_regime_tax:,.2f}
                                    """
                                    
                                    advice = get_perplexity_response(f"""Based on the following tax information, provide specific advice on tax optimization:
                                    {income_breakdown}
                                    
                                    Please provide:
                                    1. Analysis of current tax situation
                                    2. Specific recommendations for tax optimization
                                    3. Potential savings opportunities
                                    4. Any relevant tax planning strategies
                                    
                                    Format the response in a clear, structured manner.""")
                                    
                                    st.subheader("AI Tax Advisor Recommendations")
                                    st.write(advice)
                        else:
                            st.info("Please calculate your tax first to get personalized AI tax advice.")

    else:  # Upload Salary Slip
        st.subheader("Upload Salary Slip")
        st.write("Upload your salary slip in PDF or image format")
        
        # Initialize session state for form data if not exists
        if 'form_data' not in st.session_state:
            st.session_state.form_data = {
                'basic_salary': 0,
                'hra': 0,
                'special_allowance': 0,
                'bonus': 0,
                'section_80c': 0,
                'section_80d': 0,
                'home_loan_interest': 0,
                'rent_paid': 0
            }
        
        # Initialize session state for tax results if not exists
        if 'tax_results' not in st.session_state:
            st.session_state.tax_results = None
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'png', 'jpg', 'jpeg'])
        
        # Add radio button for slip type
        slip_type = st.radio("Select Slip Type", ["Monthly", "Yearly"])
        
        # Add tips for better extraction
        with st.expander("Tips for better extraction"):
            st.write("""
            - Ensure the document is clear and well-lit
            - For PDFs, use text-based PDFs rather than scanned documents
            - For images, ensure good resolution and minimal glare
            - Make sure all text is visible and not cut off
            - Avoid documents with heavy watermarks or overlays
            """)
        
        if uploaded_file is not None:
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("File Details:")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Process the uploaded file
            try:
                if uploaded_file.type == "application/pdf":
                    # Process PDF
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    # Process image
                    image = Image.open(uploaded_file)
                    text = pytesseract.image_to_string(image)
                
                # Display extracted text for debugging
                with st.expander("View Extracted Text"):
                    st.text(text)
                
                # Process the extracted text
                extracted_data = process_document_text(text)
                
                # Calculate confidence score
                confidence_score = calculate_extraction_confidence(extracted_data)
                
                # Display confidence score with color coding
                if confidence_score < 0.5:
                    st.error(f"Low confidence in extraction ({confidence_score:.0%}). Please verify the values.")
                elif confidence_score < 0.8:
                    st.warning(f"Medium confidence in extraction ({confidence_score:.0%}). Please verify the values.")
                else:
                    st.success(f"High confidence in extraction ({confidence_score:.0%})")
                
                # Create a form for the extracted values
                with st.form("salary_slip_form"):
                    st.subheader("Review and Edit Extracted Values")
                    
                    # Income Details
                    st.write("Income Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        basic_salary = st.number_input("Basic Salary", 
                            value=float(extracted_data.get('basic_salary', st.session_state.form_data['basic_salary'])),
                            min_value=0.0,
                            step=1000.0)
                        
                        hra = st.number_input("HRA", 
                            value=float(extracted_data.get('hra', st.session_state.form_data['hra'])),
                            min_value=0.0,
                            step=1000.0)
                    
                    with col2:
                        special_allowance = st.number_input("Special Allowance", 
                            value=float(extracted_data.get('special_allowance', st.session_state.form_data['special_allowance'])),
                            min_value=0.0,
                            step=1000.0)
                        
                        bonus = st.number_input("Bonus", 
                            value=float(extracted_data.get('bonus', st.session_state.form_data['bonus'])),
                            min_value=0.0,
                            step=1000.0)
                    
                    # Deductions
                    st.write("Deductions")
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        section_80c = st.number_input("Section 80C Investments", 
                            value=float(extracted_data.get('section_80c', st.session_state.form_data['section_80c'])),
                            min_value=0.0,
                            step=1000.0)
                        
                        section_80d = st.number_input("Section 80D (Health Insurance)", 
                            value=float(extracted_data.get('section_80d', st.session_state.form_data['section_80d'])),
                            min_value=0.0,
                            step=1000.0)
                    
                    with col4:
                        home_loan_interest = st.number_input("Home Loan Interest", 
                            value=float(extracted_data.get('home_loan_interest', st.session_state.form_data['home_loan_interest'])),
                            min_value=0.0,
                            step=1000.0)
                        
                        rent_paid = st.number_input("Rent Paid", 
                            value=float(extracted_data.get('rent_paid', st.session_state.form_data['rent_paid'])),
                            min_value=0.0,
                            step=1000.0)
                    
                    # Calculate button
                    submitted = st.form_submit_button("Calculate Tax")
                
                # Process form submission and show results outside the form
                if submitted:
                    # Update session state with form data
                    st.session_state.form_data = {
                        'basic_salary': basic_salary,
                        'hra': hra,
                        'special_allowance': special_allowance,
                        'bonus': bonus,
                        'section_80c': section_80c,
                        'section_80d': section_80d,
                        'home_loan_interest': home_loan_interest,
                        'rent_paid': rent_paid
                    }
                    
                    # Convert monthly values to yearly if needed
                    if slip_type == "Monthly":
                        basic_salary = basic_salary * 12
                        hra = hra * 12
                        special_allowance = special_allowance * 12
                        # Bonus is typically yearly, so no conversion needed
                        st.info("Monthly values have been converted to yearly for tax calculation")
                    
                    # Prepare data for tax calculation
                    income_details = {
                        'basic_salary': basic_salary,
                        'hra': hra,
                        'special_allowance': special_allowance,
                        'bonus': bonus,
                        'total_income': basic_salary + hra + special_allowance + bonus
                    }
                    
                    deductions = {
                        'section_80c': section_80c,
                        'section_80d': section_80d,
                        'home_loan_interest': home_loan_interest,
                        'rent_paid': rent_paid,
                        'total_deductions': section_80c + section_80d + home_loan_interest
                    }
                    
                    # Calculate tax for both regimes
                    old_regime_tax = calculate_tax(income_details, deductions, "old")
                    new_regime_tax = calculate_tax(income_details, deductions, "new")
                    
                    # Store tax results in session state
                    st.session_state.tax_results = {
                        'income_details': income_details,
                        'deductions': deductions,
                        'old_regime_tax': old_regime_tax,
                        'new_regime_tax': new_regime_tax
                    }
                
                # Display results if they exist in session state
                if st.session_state.tax_results:
                    results = st.session_state.tax_results
                    income_details = results['income_details']
                    deductions = results['deductions']
                    old_regime_tax = results['old_regime_tax']
                    new_regime_tax = results['new_regime_tax']
                    
                    # Display results
                    st.header("Tax Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Old Regime")
                        st.write(f"Total Income: ₹{income_details['total_income']:,.2f}")
                        st.write(f"Total Deductions: ₹{deductions['total_deductions']:,.2f}")
                        st.write(f"Standard Deduction: ₹50,000")
                        st.write(f"Taxable Income: ₹{income_details['total_income'] - deductions['total_deductions'] - 50000:,.2f}")
                        st.write(f"Total Tax: ₹{old_regime_tax:,.2f}")
                        
                        # Show tax calculation details
                        with st.expander("Show Tax Calculation Details (Old Regime)"):
                            st.write("Tax Slabs:")
                            st.write("Up to ₹2.5L: 0%")
                            st.write("₹2.5L–₹5L: 5%")
                            st.write("₹5L–₹10L: 20%")
                            st.write("Above ₹10L: 30%")
                            
                            # Show step-by-step calculation
                            taxable_income = max(0, income_details['total_income'] - deductions['total_deductions'] - 50000)
                            st.write(f"Taxable Income: ₹{taxable_income:,.2f}")
                            
                            if taxable_income <= 0:
                                st.write("Taxable income is zero or negative. No tax applicable.")
                            else:
                                if taxable_income <= 250000:
                                    tax = 0
                                    st.write("Tax: ₹0 (Income below ₹2.5L)")
                                elif taxable_income <= 500000:
                                    tax = 0.05 * (taxable_income - 250000)
                                    st.write(f"Tax: 5% of (₹{taxable_income:,.2f} - ₹250,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 1000000:
                                    tax = 12500 + 0.20 * (taxable_income - 500000)
                                    st.write(f"Tax: ₹12,500 + 20% of (₹{taxable_income:,.2f} - ₹500,000) = ₹{tax:,.2f}")
                                else:
                                    tax = 112500 + 0.30 * (taxable_income - 1000000)
                                    st.write(f"Tax: ₹1,12,500 + 30% of (₹{taxable_income:,.2f} - ₹10,00,000) = ₹{tax:,.2f}")
                                
                                cess = tax * 0.04
                                total_tax = tax + cess
                                st.write(f"Base Tax: ₹{tax:,.2f}")
                                st.write(f"Health & Education Cess (4%): ₹{cess:,.2f}")
                                st.write(f"Total Tax: ₹{total_tax:,.2f}")
                    
                    with col2:
                        st.subheader("New Regime")
                        st.write(f"Total Income: ₹{income_details['total_income']:,.2f}")
                        st.write(f"Standard Deduction: ₹75,000")
                        st.write(f"Taxable Income: ₹{income_details['total_income'] - 75000:,.2f}")
                        st.write(f"Total Tax: ₹{new_regime_tax:,.2f}")
                        
                        # Show tax calculation details
                        with st.expander("Show Tax Calculation Details (New Regime)"):
                            st.write("Tax Slabs:")
                            st.write("Up to ₹4L: 0%")
                            st.write("₹4L–₹8L: 5%")
                            st.write("₹8L–₹12L: 10%")
                            st.write("₹12L–₹16L: 15%")
                            st.write("₹16L–₹20L: 20%")
                            st.write("₹20L–₹24L: 25%")
                            st.write("Above ₹24L: 30%")
                            
                            # Show step-by-step calculation
                            taxable_income = max(0, income_details['total_income'] - 75000)
                            st.write(f"Taxable Income: ₹{taxable_income:,.2f}")
                            
                            if taxable_income <= 0:
                                st.write("Taxable income is zero or negative. No tax applicable.")
                            else:
                                if taxable_income <= 400000:
                                    tax = 0
                                    st.write("Tax: ₹0 (Income below ₹4L)")
                                elif taxable_income <= 800000:
                                    tax = 0.05 * (taxable_income - 400000)
                                    st.write(f"Tax: 5% of (₹{taxable_income:,.2f} - ₹400,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 1200000:
                                    tax = 20000 + 0.10 * (taxable_income - 800000)
                                    st.write(f"Tax: ₹20,000 + 10% of (₹{taxable_income:,.2f} - ₹800,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 1600000:
                                    tax = 60000 + 0.15 * (taxable_income - 1200000)
                                    st.write(f"Tax: ₹60,000 + 15% of (₹{taxable_income:,.2f} - ₹12,00,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 2000000:
                                    tax = 120000 + 0.20 * (taxable_income - 1600000)
                                    st.write(f"Tax: ₹1,20,000 + 20% of (₹{taxable_income:,.2f} - ₹16,00,000) = ₹{tax:,.2f}")
                                elif taxable_income <= 2400000:
                                    tax = 200000 + 0.25 * (taxable_income - 2000000)
                                    st.write(f"Tax: ₹2,00,000 + 25% of (₹{taxable_income:,.2f} - ₹20,00,000) = ₹{tax:,.2f}")
                                else:
                                    tax = 300000 + 0.30 * (taxable_income - 2400000)
                                    st.write(f"Tax: ₹3,00,000 + 30% of (₹{taxable_income:,.2f} - ₹24,00,000) = ₹{tax:,.2f}")
                                
                                cess = tax * 0.04
                                total_tax = tax + cess
                                st.write(f"Base Tax: ₹{tax:,.2f}")
                                st.write(f"Health & Education Cess (4%): ₹{cess:,.2f}")
                                st.write(f"Total Tax: ₹{total_tax:,.2f}")
                    
                    # Recommendation
                    st.header("Recommendation")
                    if old_regime_tax < new_regime_tax:
                        selected_regime = "Old Regime"
                        st.success(f"The Old Regime is better for you! You'll save ₹{new_regime_tax - old_regime_tax:,.2f}")
                    else:
                        selected_regime = "New Regime"
                        st.success(f"The New Regime is better for you! You'll save ₹{old_regime_tax - new_regime_tax:,.2f}")
                    
                    # Add AI tax advice button outside the form
                    if st.button("Get AI Tax Advice"):
                        with st.spinner("Getting personalized tax advice..."):
                            # Format the data for the prompt
                            income_breakdown = f"""
                            Income Details:
                            - Basic Salary: ₹{income_details['basic_salary']:,.2f}
                            - HRA: ₹{income_details['hra']:,.2f}
                            - Special Allowance: ₹{income_details['special_allowance']:,.2f}
                            - Bonus: ₹{income_details['bonus']:,.2f}
                            - Total Income: ₹{income_details['total_income']:,.2f}
                            
                            Deductions:
                            - Section 80C: ₹{deductions['section_80c']:,.2f}
                            - Section 80D: ₹{deductions['section_80d']:,.2f}
                            - Home Loan Interest: ₹{deductions['home_loan_interest']:,.2f}
                            - Rent Paid: ₹{deductions['rent_paid']:,.2f}
                            - Total Deductions: ₹{deductions['total_deductions']:,.2f}
                            
                            Tax Amounts:
                            - Old Regime Tax: ₹{old_regime_tax:,.2f}
                            - New Regime Tax: ₹{new_regime_tax:,.2f}
                            """
                            
                            advice = get_perplexity_response(f"""Based on the following tax information, provide specific advice on tax optimization:
                            {income_breakdown}
                            
                            Please provide:
                            1. Analysis of current tax situation
                            2. Specific recommendations for tax optimization
                            3. Potential savings opportunities
                            4. Any relevant tax planning strategies
                            
                            Format the response in a clear, structured manner.""")
                            
                            st.subheader("AI Tax Advisor Recommendations")
                            st.write(advice)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 