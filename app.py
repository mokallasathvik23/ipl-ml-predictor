import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# 🏏 Page Config
# ----------------------------
st.set_page_config(page_title="IPL Predictor", layout="wide")

st.title("🏏 IPL Ultimate Match Predictor")

# ----------------------------
# 📂 Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("matches.csv")
    df = df.dropna(subset=['winner'])
    df = df[['team1','team2','winner','toss_winner','toss_decision','venue']]
    return df

matches = load_data()

# ----------------------------
# 🔤 Encode Teams
# ----------------------------
le = LabelEncoder()
matches['team1'] = le.fit_transform(matches['team1'])
matches['team2'] = le.fit_transform(matches['team2'])
matches['winner'] = le.fit_transform(matches['winner'])

# ----------------------------
# 🎯 Feature Engineering
# ----------------------------
matches['toss_win'] = (matches['toss_winner'] == matches['winner']).astype(int)

X = matches[['team1','team2','toss_win']]
y = matches['winner']

# ----------------------------
# 🤖 Train Model
# ----------------------------
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model(X, y)

# ----------------------------
# 🎮 Sidebar
# ----------------------------
st.sidebar.title("About")
st.sidebar.info("Advanced IPL match prediction using Machine Learning.")

# ----------------------------
# 🎮 UI Layout
# ----------------------------
teams = list(le.classes_)
venues = matches['venue'].unique()

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", teams)

with col2:
    team2 = st.selectbox("Select Team 2", teams)

if team1 == team2:
    st.warning("⚠️ Select different teams")
    st.stop()

# Toss + Venue
col3, col4 = st.columns(2)

with col3:
    toss = st.selectbox("Who won the toss?", [team1, team2])

with col4:
    venue = st.selectbox("Select Venue", venues)

# ----------------------------
# 📊 Helper Functions
# ----------------------------
def head_to_head(team1, team2):
    matches_h2h = matches[
        ((matches['team1']==team1) & (matches['team2']==team2)) |
        ((matches['team1']==team2) & (matches['team2']==team1))
    ]
    return len(matches_h2h)

def recent_form(team):
    last_matches = matches.tail(50)
    wins = len(last_matches[last_matches['winner'] == team])
    return wins

# ----------------------------
# 🔮 Prediction
# ----------------------------
if st.button("🚀 Predict Winner"):

    with st.spinner("Analyzing match..."):
        time.sleep(1)

        t1 = le.transform([team1])[0]
        t2 = le.transform([team2])[0]

        toss_win = 1 if toss == team1 else 0

        input_data = np.array([[t1, t2, toss_win]])

        prediction = model.predict(input_data)
        winner = le.inverse_transform(prediction)[0]

        proba = model.predict_proba(input_data)[0]
        team_probs = dict(zip(le.classes_, proba))

    # ----------------------------
    # 📊 Output
    # ----------------------------
    st.success(f"🏆 Predicted Winner: {winner}")

    st.subheader("📊 Winning Probability")
    st.write(f"🔵 {team1}: {round(team_probs.get(team1,0)*100,2)}%")
    st.write(f"🔴 {team2}: {round(team_probs.get(team2,0)*100,2)}%")

    confidence = max(team_probs.values()) * 100
    st.write(f"🔥 Confidence: {round(confidence,2)}%")

    # ----------------------------
    # 📈 Insights
    # ----------------------------
    st.subheader("📈 Match Insights")

    h2h_matches = head_to_head(t1, t2)
    st.write(f"🤝 Head-to-Head Matches: {h2h_matches}")

    form1 = recent_form(t1)
    form2 = recent_form(t2)

    st.write(f"📊 Recent Form:")
    st.write(f"{team1}: {form1} wins (recent)")
    st.write(f"{team2}: {form2} wins (recent)")

    st.write("✔ Toss impact considered")
    st.write("✔ Historical trends used")

# ----------------------------
# 🔄 Reset
# ----------------------------
if st.button("Reset"):
    st.rerun()