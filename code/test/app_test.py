import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

import json
json_data = {"customer_id":{"0":1001,"1":1002,"2":1003,"3":1004},"Account_Balance":{"0":15000,"1":32000,"2":-5000,"3":70000},"Transaction_Amount":{"0":500,"1":1200,"2":300,"3":2000},"Reported_Amount":{"0":500,"1":1200,"2":300,"3":1800},"Currency":{"0":"USD","1":"EUR","2":"GBP","3":"USD"},"Country":{"0":"US","1":"DE","2":"UK","3":"US"},"Transaction_Date":{"0":"2025-02-25","1":"2025-02-20","2":"2025-02-18","3":"2025-02-28"},"Risk_Score":{"0":3,"1":2,"2":6,"3":5}}
json_str = json.dumps(json_data)
df = pd.read_json(json_str)


# Define a function to validate the transaction amount vs. reported amount
def validate_amount(row):
    if row["Transaction_Amount"] != row["Reported_Amount"]:
        # Check if the transaction involves cross-currency conversions
        if row["Currency"] != "USD":
            # Allow a permissible deviation of up to 1%
            if abs(row["Transaction_Amount"] - row["Reported_Amount"]) / row["Transaction_Amount"] <= 0.01:
                return True
            else:
                return False
        else:
            return False
    else:
        return True

# Define a function to validate the account balance
def validate_balance(row):
    if row["Account_Balance"] < 0:
        # Check if the account is an overdraft account
 #       if "OD" in row["customer_id"]:
 #           return True
 #       else:
            return True
    else:
        return True

# Define a function to validate the currency
def validate_currency(row):
    # Check if the currency is a valid ISO 4217 currency code
    valid_currencies = ["USD", "EUR", "GBP"]
    if row["Currency"] not in valid_currencies:
        return False
    else:
        return True

# Define a function to validate the country
def validate_country(row):
    # Check if the country is an accepted jurisdiction
    accepted_jurisdictions = ["US", "DE", "UK"]
    if row["Country"] not in accepted_jurisdictions:
        return False
    else:
        return True

# Define a function to validate the transaction date
def validate_date(row):
    # Check if the transaction date is not in the future
    if datetime.strptime(row["Transaction_Date"], "%Y-%m-%d") > datetime.now():
        return False
    else:
        return True

# Define a function to validate high-risk transactions
def validate_high_risk(row):
    # Check if the transaction amount is greater than $5,000 in high-risk countries
    high_risk_countries = ["US"]
    if row["Transaction_Amount"] > 5000 and row["Country"] in high_risk_countries:
        return False
    else:
        return True

# Define a function to validate round-number transactions
def validate_round_number(row):
    # Check if the transaction amount is a round number
    if row["Transaction_Amount"] % 1000 == 0:
        return False
    else:
        return True

# Apply the validation functions to the dataset
df["Amount_Valid"] = df.apply(validate_amount, axis=1)
df["Balance_Valid"] = df.apply(validate_balance, axis=1)
df["Currency_Valid"] = df.apply(validate_currency, axis=1)
df["Country_Valid"] = df.apply(validate_country, axis=1)
df["Date_Valid"] = df.apply(validate_date, axis=1)
df["High_Risk_Valid"] = df.apply(validate_high_risk, axis=1)
df["Round_Number_Valid"] = df.apply(validate_round_number, axis=1)

# Implement dynamic risk scoring system
def calculate_risk_score(row):
    # Calculate risk score based on transaction patterns and historical violations
    return 0

df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)

# Trigger compliance checks for high-risk transactions
def trigger_compliance_check(row):
    # Trigger compliance check for high-risk transactions
    return False


df['Compliance_Check'] = df.apply(trigger_compliance_check, axis=1)

# Convert DataFrame to readable text for LLM
context_text = "Dataset Validation Report:\n"
context_text += "Columns: customer_id, Account_Balance, Transaction_Amount,Reported_Amount,Currency,Country,Transaction_Date, Risk_Score,Amount_Valid,Balance_Valid,Currency_Valid,Country_Valid,Date_Valid,High_Risk_Valid,Round_Number_Valid\n"
context_text += "\n".join([
    f"{row['customer_id']}, {row['Account_Balance']}, {row['Transaction_Amount']},{row['Reported_Amount']},{row['Currency']},{row['Country']},{row['Transaction_Date']},{row['Risk_Score']},{row['Amount_Valid']},{row['Balance_Valid']},{row['Currency_Valid']},{row['Country_Valid']},{row['Date_Valid']},{row['High_Risk_Valid']},{row['Round_Number_Valid']}"
    for _, row in df.iterrows()
])

#print(context_text)


# Create dynamic risk scoring model
#X = df.drop(['Risk_Score'], axis=1)
#print(X)
#y = df['Risk_Score']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.dtypes)
#print(X_train.head())

#X_train.fillna(0, inplace=True)
#X_test.fillna(0, inplace=True)

#y_train.fillna(0, inplace=True)
#y_test.fillna(0, inplace=True)

#model = LogisticRegression()
#model.fit(X_train, y_train)

# Suggest remediation actions for flagged transactions
def suggest_remediation_actions(row):
    if row['Risk_Score'] > 5:
        # Flagged transaction
        return 'Adjust transaction amount to $500'
    return 'No action required'

# Define queries for validation
queries = [
    "Transaction_Amount should always match reported_amount, expect when the transaction involves cross-currency conversions, in which case a permissible deviation of up to 1% is allowed ",
	"Account_balance should never be negative,except in case of overdraft accounts explicitly marked with an 'OD' flag",
	"Currency should be a valid ISO 4217 currency code, and the transaction must adhere to cross-border transaction limits as per regulatory guidelines.",
	"Country should be an accepted jurisdiction based on bank regulations, and cross-border transactions should include mandatory transaction remarks if the amount exceeds $10,000."
	"Transaction_date should not be in the future, and transactions older than 365 days should trigger a data validation alert.",
	"High-risk transactions (amount > $5,000 in high-risk countries) should be flagged with an automatic compliance check triggered",
	"Round-number transactions(e.g., $1000,$5000) should be analyzed for potential money laundering risks, requiring additional validations steps.",
	"A Dynamic risk scoring system should be implemented, adjusting scores based on transactions patterns and historical violations."
	"The model should suggest remediation actions for flagged transactions, including adjustments. explanations, and recommended compliance steps."
	]


# Together AI API Key
TOGETHER_AI_KEY = "4b257a228bfa9595d47d41544c7ec470ec5ed4e8556e812e070129559d4e8b0f"

# Send request to Together AI (Claude 3)
response = requests.post(
    "https://api.together.xyz/v1/chat/completions",
    headers={"Authorization": f"Bearer {TOGETHER_AI_KEY}", "Content-Type": "application/json"},
    json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", "messages": [{"role": "user", "content": f"Dataset:\n{context_text}\nQueries:\n" + "\n".join(queries)}]}
)

# Print the full API response for debugging
print("\n🔍 API Raw Response:", response.json())


# Get response and print validation results
result = response.json()["choices"][0]["message"]["content"]
print("\n **Validation Report:**\n", result)



