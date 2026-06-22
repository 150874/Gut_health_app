import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# 1. Load your original dataset
df_original = pd.read_excel(r"C:\Users\IOrimba\OneDrive - GardaWorld\Desktop\GUT HEALTH\comprehensive_gut_health_ml_dataset.xlsx", sheet_name='Sheet1')

# 2. Define the baseline features we want to KEEP for the ML model
baseline_cols = [
    'User_ID', 'Age', 'BMI', 'Primary_Condition', 'H_Pylori_Test_Result', 
    'Diet_Pattern', 'Stress_Level', 'NSAID_Use'
]
df_baseline = df_original[baseline_cols].copy()

# 3. Create the Event Generator Logic
# We will simulate 3 days of meal logs (Breakfast, Lunch, Dinner) for each user
meal_types = ['Breakfast', 'Lunch', 'Dinner']
logs = []

print("Optimizing dataset into event-based logs...")

for index, user in df_baseline.iterrows():
    # Simulate 3 days of tracking per user to give the ML time-series data
    start_date = datetime(2023, 10, 1)
    
    for day_offset in range(3):
        current_date = start_date + timedelta(days=day_offset)
        
        for meal in meal_types:
            # Assign a realistic timestamp
            if meal == 'Breakfast':
                meal_time = current_date.replace(hour=random.randint(7, 9), minute=random.randint(0, 59))
            elif meal == 'Lunch':
                meal_time = current_date.replace(hour=random.randint(12, 14), minute=random.randint(0, 59))
            else:
                meal_time = current_date.replace(hour=random.randint(18, 21), minute=random.randint(0, 59))
            
            # Simulate PRAL Score based on their Diet Pattern
            # Western diets get higher (more acidic) PRAL, Vegan gets lower (alkaline)
            if user['Diet_Pattern'] == 'Western':
                pral_score = round(np.random.normal(5.0, 3.0), 1) 
            elif user['Diet_Pattern'] == 'Vegan' or user['Diet_Pattern'] == 'Vegetarian':
                pral_score = round(np.random.normal(-2.0, 2.0), 1)
            else:
                pral_score = round(np.random.normal(1.0, 2.5), 1)
                
            # Simulate Water consumed with this specific meal (in ml)
            water_ml = random.choice([0, 200, 250, 500])
            
            # --- THE TARGET VARIABLE (What the ML will predict) ---
            # We simulate a "Flare-Up Score" (0-10) 3 hours after eating
            # High PRAL + High Stress + Low Water = Higher risk of flare up
            flare_risk = 0
            if pral_score > 3.0: flare_risk += 3
            if water_ml < 200: flare_risk += 2
            if user['Stress_Level'] == 'High': flare_risk += 2
            if user['Primary_Condition'] == 'GERD' and meal == 'Dinner': flare_risk += 2 # Late night GERD
            
            # Add some randomness so the ML has to work for it
            actual_flare_score = min(10, max(0, flare_risk + random.randint(-2, 2)))
            
            # Append the event
            logs.append({
                'User_ID': user['User_ID'],
                'Age': user['Age'],
                'BMI': user['BMI'],
                'Primary_Condition': user['Primary_Condition'],
                'H_Pylori_Result': user['H_Pylori_Test_Result'],
                'Timestamp': meal_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Meal_Type': meal,
                'Meal_PRAL_Score': pral_score,
                'Water_Consumed_ml': water_ml,
                'Stress_At_Meal': user['Stress_Level'],
                'Symptom_Flare_Up_Score': actual_flare_score # THIS IS YOUR Y-VARIABLE
            })

# 4. Convert the logs into a new DataFrame
df_optimized = pd.DataFrame(logs)

# 5. Save the optimized dataset for Machine Learning
output_filename = 'optimized_meal_logs_dataset.csv'
df_optimized.to_csv(output_filename, index=False)

print(f"Success! Optimized dataset saved as: {output_filename}")
print(f"Expanded {len(df_original)} users into {len(df_optimized)} individual meal events.")