from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

app = Flask(__name__)

# --- Load and clean the dataset once ---
df = pd.read_excel("bank.xlsx")
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
df['VALUE DATE'] = pd.to_datetime(df['VALUE DATE'], errors='coerce')
df['WITHDRAWAL AMT'] = pd.to_numeric(df['WITHDRAWAL AMT'], errors='coerce').fillna(0)
df['DEPOSIT AMT'] = pd.to_numeric(df['DEPOSIT AMT'], errors='coerce').fillna(0)
df['Month'] = df['DATE'].dt.to_period('M')

monthly_summary = df.groupby('Month').agg({
    'DEPOSIT AMT': 'sum',
    'WITHDRAWAL AMT': 'sum'
}).reset_index()
monthly_summary['Savings'] = monthly_summary['DEPOSIT AMT'] - monthly_summary['WITHDRAWAL AMT']

features = pd.DataFrame({
    'avg_savings': monthly_summary['Savings'].rolling(window=3).mean().fillna(method='bfill'),
    'savings_volatility': monthly_summary['Savings'].rolling(window=3).std().fillna(0),
    'avg_deposit': monthly_summary['DEPOSIT AMT'].rolling(window=3).mean().fillna(method='bfill'),
    'avg_withdrawal': monthly_summary['WITHDRAWAL AMT'].rolling(window=3).mean().fillna(method='bfill'),
})

# --- KMeans risk profiling ---
def infer_risk_profile():
    X = StandardScaler().fit_transform(features.fillna(0).values)
    best_k, best_score, best_km = None, -1, None
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_km = k, score, km
    latest_cluster = best_km.predict(X[-1].reshape(1, -1))[0]
    centres = StandardScaler().fit(features.fillna(0).values).inverse_transform(best_km.cluster_centers_)
    vol_rank = np.argsort(centres[:, 1])
    risk_labels = {
        vol_rank[0]: "Conservative",
        vol_rank[len(vol_rank)//2]: "Balanced",
        vol_rank[-1]: "Aggressive"
    }
    return risk_labels.get(latest_cluster, "Balanced")

# --- Citi Investment Products ---
investment_universe = [
    {
        "name": "Citi India BlueChip Fund",
        "type": "Equity",
        "risk": "Conservative",
        "expected_return": 6.5,
        "horizon": 3,
        "liquidity": "Medium"
    },
    {
        "name": "Citi Global Growth Portfolio",
        "type": "Equity",
        "risk": "Aggressive",
        "expected_return": 13,
        "horizon": 7,
        "liquidity": "Low"
    },
    {
        "name": "Citi Government Securities Fund",
        "type": "Bonds",
        "risk": "Conservative",
        "expected_return": 4.2,
        "horizon": 2,
        "liquidity": "High"
    },
    {
        "name": "Citi Corporate Bond Opportunities",
        "type": "Bonds",
        "risk": "Balanced",
        "expected_return": 6.8,
        "horizon": 4,
        "liquidity": "Medium"
    },
    {
        "name": "Citi Gold Savings Plan",
        "type": "Gold",
        "risk": "Balanced",
        "expected_return": 5.5,
        "horizon": 3,
        "liquidity": "High"
    },
    {
        "name": "Citi Digital Innovation Fund",
        "type": "Equity",
        "risk": "Aggressive",
        "expected_return": 14.5,
        "horizon": 8,
        "liquidity": "Low"
    },
    {
        "name": "Citi Ultra Short-Term Debt Fund",
        "type": "Bonds",
        "risk": "Conservative",
        "expected_return": 4.8,
        "horizon": 1,
        "liquidity": "High"
    },
    {
        "name": "Citi ESG Opportunity Fund",
        "type": "Equity",
        "risk": "Balanced",
        "expected_return": 8.5,
        "horizon": 5,
        "liquidity": "Medium"
    },
    {
        "name": "Citi Diversified Income Plan",
        "type": "Hybrid",
        "risk": "Balanced",
        "expected_return": 7.0,
        "horizon": 4,
        "liquidity": "Medium"
    }
]


# --- Recommend based on user input ---
def recommend_investments(user_input, risk):
    preferred_types = [t.lower() for t in user_input["investment_type"]]
    min_return = user_input["expected_return_pct"]
    horizon = user_input["investment_horizon_years"]
    liquidity = user_input["liquidity_need"].lower()

    filtered = [
        inv for inv in investment_universe
        if inv['risk'].lower() == risk.lower()
        and inv['type'].lower() in preferred_types
        and inv['expected_return'] >= min_return
    ]

    if not filtered:
        filtered = [
            inv for inv in investment_universe
            if inv['risk'].lower() == risk.lower()
            and inv['type'].lower() in preferred_types
        ]

    recommendations = []
    for inv in filtered:
        notes = []
        if inv["horizon"] > horizon:
            notes.append(f"Horizon mismatch (needs {inv['horizon']} yrs)")
        if inv["liquidity"].lower() != liquidity:
            notes.append(f"Liquidity is {inv['liquidity']} (you prefer {liquidity})")
        recommendations.append({
            "name": inv["name"],
            "type": inv["type"],
            "risk": inv["risk"],
            "expected_return": inv["expected_return"],
            "horizon": inv["horizon"],
            "liquidity": inv["liquidity"],
            "notes": notes
        })

    return recommendations

# --- Flask route ---
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json
    risk = infer_risk_profile()
    recs = recommend_investments(user_input, risk)
    return jsonify({
        "inferred_risk_profile": risk,
        "recommendations": recs
    })

@app.route("/", methods=["GET"])
def forward_form():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
