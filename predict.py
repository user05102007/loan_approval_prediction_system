import joblib
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load saved files
model = joblib.load("models/best_loan_approval_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

print(Fore.CYAN + "=" * 60)
print(Fore.CYAN + "        LOAN APPROVAL PREDICTION SYSTEM")
print(Fore.CYAN + "=" * 60)

# User Inputs
dependents = int(input("Enter Number of Dependents: "))

education = input(
    "Education (Graduate/Not Graduate): "
).strip().lower()

self_emp = input(
    "Self Employed (Yes/No): "
).strip().lower()

income = float(input("Annual Income: "))
loan_amount = float(input("Loan Amount: "))
loan_term = float(input("Loan Term (Months): "))
cibil = float(input("CIBIL Score: "))
residential = float(input("Residential Assets Value: "))
commercial = float(input("Commercial Assets Value: "))
luxury = float(input("Luxury Assets Value: "))
bank = float(input("Bank Asset Value: "))

# Encoding
education = 0 if education == "graduate" else 1
self_emp = 1 if self_emp == "yes" else 0

# Create DataFrame
data = pd.DataFrame([[
    dependents,
    education,
    self_emp,
    income,
    loan_amount,
    loan_term,
    cibil,
    residential,
    commercial,
    luxury,
    bank
]], columns=[
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
])

# Scale Data
data_scaled = scaler.transform(data)

# Prediction
prediction = model.predict(data_scaled)[0]

# Probability
if hasattr(model, "predict_proba"):
    probability = max(model.predict_proba(data_scaled)[0]) * 100
else:
    probability = 0

# Decode Result
status = encoder.inverse_transform([prediction])[0]

# Analysis
reasons = []
total_assets = residential + commercial + luxury + bank

if cibil >= 750:
    reasons.append("✔ Excellent CIBIL Score")

elif cibil >= 650:
    reasons.append("✔ Good CIBIL Score")

else:
    reasons.append("✘ Low CIBIL Score")

if income > loan_amount:
    reasons.append("✔ Income is higher than loan amount")
else:
    reasons.append("✘ Loan amount is high compared to income")

if total_assets > loan_amount:
    reasons.append("✔ Strong asset backing")
else:
    reasons.append("✘ Low asset coverage")

print("\n" + "=" * 60)

if "approved" in status.lower():

    print(Fore.GREEN + Style.BRIGHT +
          f"LOAN STATUS : {status.upper()}")

    print(Fore.GREEN +
          f"CONFIDENCE  : {probability:.2f}%")

    print(Fore.GREEN +
          "RISK LEVEL  : LOW")

else:

    print(Fore.RED + Style.BRIGHT +
          f"LOAN STATUS : {status.upper()}")

    print(Fore.RED +
          f"CONFIDENCE  : {probability:.2f}%")

    print(Fore.RED +
          "RISK LEVEL  : HIGH")

print("\nREASONS:")
for reason in reasons:
    print(reason)

print("=" * 60)