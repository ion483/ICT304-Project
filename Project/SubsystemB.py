import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib
import os
from datetime import datetime
import argparse
import random

random.seed(42)
np.random.seed(42)

item_profiles = {}
item_names = {0:'Frontdoor keys', 1:'Phone', 2:'Toothpaste', 3:'Wrist watch'}
        
      
if os.path.exists('mock_sales_history.csv'):
    print("--- Sales history found ---")
else:
    print("--- Creating mock sales history ---")

    items = {
        0: {'name': 'Frontdoor keys', 'base': 5, 'weekend_boost': 1.6},
        1: {'name': 'Phone', 'base': 15, 'weekend_boost': 1.3},
        2: {'name': 'Toothpaste', 'base': 12, 'weekend_boost': 1.4},
        3: {'name': 'Wrist watch', 'base': 8, 'weekend_boost': 1.5}
    }

    dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
    sales_data = []

    for date in dates:
        for item_id, info in items.items():
            if date.weekday() >= 5:
                sales = int(info['base'] * info['weekend_boost'] * np.random.uniform(0.9, 1.2))
            else:
                sales = int(info['base'] * np.random.uniform(0.9, 1.1))
            
            sales_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'item_id': item_id,
                'item_name': info['name'],
                'quantity_sold': max(0, sales)
            })

    pd.DataFrame(sales_data).to_csv('mock_sales_history.csv', index=False)
    print("mock_sales_history.csv created")



""""""


def extract_features(sales_df, item_id):
    """Create features for prediction"""
    item_sales = sales_df[sales_df['item_id'] == item_id].sort_values('date')
    
    if len(item_sales) < 14:
        return None, None
    
    X, y = [], []
    for i in range(7, len(item_sales)):
        last_week = item_sales.iloc[i-7:i]['quantity_sold'].values
        is_weekend = 1 if pd.to_datetime(item_sales.iloc[i]['date']).weekday() >= 5 else 0
        
        features = [
            last_week.mean(),
            last_week.std(),
            last_week[-1],
            pd.to_datetime(item_sales.iloc[i]['date']).weekday(),
            is_weekend,
            last_week[0],
            max(last_week) - min(last_week)
        ]
        
        X.append(features)
        y.append(item_sales.iloc[i]['quantity_sold'])
    
    return np.array(X), np.array(y)



def train_models():
    """Train and save models"""
    print("\n--- Training prediction models ---")
    
    sales = pd.read_csv('mock_sales_history.csv')
    
    for item_id in range(4):
        item_name = item_names[item_id]
        
        X, y = extract_features(sales, item_id)
        if X is None or len(X) < 10:
            print(f"Insufficient data for {item_name}")
            continue
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        rf = RandomForestRegressor(
            n_estimators=100,  # More trees (50 -> 100)
            max_depth=None,    # Let trees grow deeper to find complex patterns
            min_samples_leaf=2,# Prevents overfitting by requiring 2 samples per leaf
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        svm = SVR(
            kernel='rbf', 
            C=10.0, 
            gamma='scale', 
            epsilon=0.01
        )
        svm.fit(X_train, y_train)
        
        rf_score = r2_score(y_test, rf.predict(X_test))
        svm_score = r2_score(y_test, svm.predict(X_test))
        
        item_profiles[item_id] = {
            'name': item_name,
            'model': rf if rf_score > svm_score else svm,
            'type': 'Random Forest' if rf_score > svm_score else 'SVM',
            'score': max(rf_score, svm_score)
        }
        
        print(f"{item_name}: {item_profiles[item_id]['type']} (R2 = {item_profiles[item_id]['score']:.3f})")
    
    joblib.dump(item_profiles, 'item_models.pkl')
    print("\nModels saved to item_models.pkl")


""""""


def load_models():
    """Load saved models"""
    global item_profiles
    if os.path.exists('item_models.pkl'):
        item_profiles = joblib.load('item_models.pkl')
        print("--- Existing models loaded ---")
        return True
    return False



def predict_next_3_days(item_id, current_stock):
    """Predict demand and days until stockout"""
    if item_id not in item_profiles:
        return None
    
    sales = pd.read_csv('mock_sales_history.csv')
    item_sales = sales[sales['item_id'] == item_id]['quantity_sold'].tail(14).tolist()
    
    if len(item_sales) < 7:
        return None
    
    model = item_profiles[item_id]['model']
    predictions = []
    
    temp_sales = item_sales.copy()
    for day in range(3):
        date_to_predict = datetime.now() + pd.Timedelta(days = day)
        weekday = date_to_predict.weekday()
        is_weekend = 1 if weekday >= 5 else 0

        features = np.array([[
            np.mean(temp_sales[-7:]),
            np.std(temp_sales[-7:]),
            temp_sales[-1],
            weekday,
            is_weekend,
            temp_sales[-7],
            max(temp_sales[-7:]) - min(temp_sales[-7:])
        ]])
        
        pred = max(0, int(model.predict(features)[0]))
        predictions.append(pred)
        temp_sales.append(pred)
    
    total_next_three_days = sum(predictions)
    avg_daily = sum(predictions) / 3
    
    return {
        'predictions': predictions,
        'avg_daily': round(avg_daily, 1),
        'sum': total_next_three_days,
        'confidence': round(item_profiles[item_id]['score'] * 100, 1)
    }


def generate_forecast():
    """Create forecast from CurrentStocks.csv"""
    if not os.path.exists('CurrentStocks.csv'):
        print("CurrentStocks.csv not found. Run subsystemA.py first.")
        return None
    
    current_stock = pd.read_csv('CurrentStocks.csv')
    
    print("\n--- DeepTrack-Forecast Results ---")
    print("3-Day Inventory Forecast")
    print("-" * 40)
    
    forecast_data = []
    
    for _, row in current_stock.iterrows():
        try:
            item_id = int(float(row['StockCode']))
            current_qty = int(row['Quantity'])
            item_name = item_names.get(item_id, f"Item {item_id}")
            
            result = predict_next_3_days(item_id, current_qty)
           
            
            if result:
                print(f"\n{item_name}:")
                print(f"  Current: {current_qty} units")
                print(f"  Next 3 days: {result['predictions']}")
                print(f"  Avg daily: {result['avg_daily']}")
                print(f"  Sum: {result['sum']}")
                print(f"  Confidence: {result['confidence']}%")
                
                forecast_data.append({
                    'item_id': item_id,
                    'item_name': item_name,
                    'current_stock': current_qty,
                    'day1_prediction': result['predictions'][0],
                    'day2_prediction': result['predictions'][1],
                    'day3_prediction': result['predictions'][2],
                    'avg_daily_demand': result['avg_daily'],
                    'sum': result['sum'],
                    'confidence': result['confidence'],
                    'forecast_date': datetime.now().strftime('%m/%d/%Y %H:%M')
                })
            else:
                print(f"\nInsufficient data for {item_name}")
                
        except Exception as e:
            print(f"Error processing row: {e}")
    
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        forecast_df.to_csv('ForecastReport.csv', index=False)
        print(f"\nForecastReport.csv updated")
        return forecast_df
    
    return None


parser = argparse.ArgumentParser(description='DeepTrack-Forecast')
parser.add_argument('--retrain', action='store_true', help='Force retrain models')
args = parser.parse_args()

if args.retrain or not load_models():
    print("Training new models...")
    train_models()


forecast = generate_forecast()

if forecast is not None:
    print("\n--- DeepTrack-Forecast complete ---")
else:
    print("\n--- DeepTrack-Forecast failed: Run subsystemA.py first ---")