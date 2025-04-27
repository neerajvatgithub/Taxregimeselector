# Tax Regime Selector

A Streamlit application that helps Indian salaried individuals compare their tax liability under the old and new tax regimes to make an informed decision about which regime to choose.

## Features

- Manual entry of salary and deduction details
- Automatic tax calculation for both old and new regimes
- Clear comparison and recommendation
- Support for various deductions including:
  - HRA
  - Section 80C
  - Section 80D
  - Home Loan Interest
  - Standard Deduction

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Choose your preferred input method:
   - Manual Entry
   - Upload Form 16 (coming soon)
   - Upload Salary Slip (coming soon)

4. Enter your income and deduction details

5. Click "Calculate Tax" to see the comparison and recommendation

## Tax Calculation Details

### Old Regime
- Standard Deduction: ₹50,000
- All eligible exemptions and deductions apply
- Tax Slabs:
  - Up to ₹2.5L: Nil
  - ₹2.5L–₹5L: 5%
  - ₹5L–₹10L: 20%
  - Above ₹10L: 30%

### New Regime
- Standard Deduction: ₹75,000
- Limited exemptions and deductions
- Tax Slabs:
  - Up to ₹4L: Nil
  - ₹4L–₹8L: 5%
  - ₹8L–₹12L: 10%
  - ₹12L–₹16L: 15%
  - ₹16L–₹20L: 20%
  - ₹20L–₹24L: 25%
  - Above ₹24L: 30%

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 