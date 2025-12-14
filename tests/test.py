import sys
import requests
import json
import time

def test_prediction(url):
    print(f"🚀 Testing API at: {url}")
    
    # Endpoint for prediction
    predict_url = f"{url}/predict"
    
    # Sample payload matching your model's expected input
    payload = {
        "Age": 35, "Income_Annual": 75000, "Base_Salary_PerMonth": 6250.0,
        "Total_Bank_Accounts": 3, "Total_Credit_Cards": 4, "Rate_Of_Interest": 15.0,
        "Delay_from_due_date": 5, "Total_Delayed_Payments": 2, "Total_Current_Loans": 3,
        "Credit_Limit": 25000, "Total_Credit_Enquiries": 3, "Current_Debt_Outstanding": 5000,
        "Ratio_Credit_Utilization": 0.3, "Credit_History_Age": 66, "Per_Month_EMI": 500.0,
        "Credit_Mix": "Standard", "Monthly_Investment": 200,
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
        "Monthly_Balance": 1500, "Payment_of_Min_Amount": "Yes"
    }

    headers = {'Content-Type': 'application/json'}
    
    # Retry logic (in case pod is still starting up)
    max_retries = 5
    for i in range(max_retries):
        try:
            print(f"Attempt {i+1}...")
            response = requests.post(predict_url, data=json.dumps(payload), headers=headers, timeout=5)
            
            if response.status_code == 200:
                print("✅ Success! API returned 200 OK")
                print("Response:", response.json())
                return True
            else:
                print(f"⚠️ Failed with status {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"⚠️ Connection error: {e}")
        
        time.sleep(5) # Wait before retrying

    print("❌ API Test Failed after multiple attempts")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python api_test.py <base_url>")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    test_prediction(base_url)
