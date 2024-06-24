#!/usr/bin/env python
# coding: utf-8

# # Predictive Analysis of Football Outcomes with API-Football

# The purpose of this Jupyter notebook is to explore whether if we can predict the winner of a football match in the English Premier League (EPL).
# 
# Our predictive edge is that in every match, teams are influenced by psychological factors such as "Positive Momentum", where a team is on such a long winning streak, it has a self-fulfilling prophetic effect where the said team benefits from the positive energy yielded from their fans and from other team members; and in return has a negative effect on the opposing team. Conversely, we predict that a team that is on a losing streak will more likely to lose their next game due to "negative momentum". 
# 
# This is just one example of psychological effect that we will take advatange of to predict whether this variable can predict a team's victory or loss. 
# 
# We will be to performing feature engineering from raw data we obtain from API-Football such as Goals Scored by Team, Shots on Target, Poession Percentage, and Number of Coners/fouls, to create synthetic variables such as "Goal Conversion Rate" and "Defense Strength" that will help us predict a team victory/loss.

# Below are the the general steps we will follow for this project: 
# 
# 1. Download data from API-football for player and team statistics.
# 
# 2. Extract, Transform, and Load (ETL) with pandas and numpy to clean and construct data for feature engineering. 
# 
# 3. Analyze the data to identify factors that might influence soccer match outcomes, such as team form, head-to-head history, player injuries, etc to create new features in your dataset that capture these factors.
# 
# 4. Implement logistic regression and random forest classifiers using scikit-learn, train models on historical data, tuning parameters as necessary.
# 
# 5. Model Evaluation and Validation: Perform cross-validation to test the model's performance on unseen data.
# 

# The synthetically created features we will use are: 
# 
# 1. Goal Conversion Rate: Create a feature that measures the efficiency of shots taken.
# 
# - data['goal_conversion_rate'] = data['goals_scored'] / data['shots_on_target']
# 
# 
# 2. Defense Strength: A metric that could represent how strong a team's defense is, based on fouls committed and cards received, indicating aggressive defense.
# 
# - data['defense_strength'] = data['fouls_committed'] + (data['yellow_cards'] * 2) + (data['red_cards'] * 3)
# 
# 
# 3. Attacking Dominance: A feature combining possession with shots on target to reflect how much a team is dominating in attack.
# 
# - data['attacking_dominance'] = (data['possession_percentage'] * data['shots_on_target']) / 100
# 
# 4. Pressure Index: An index reflecting how much pressure a team applies, derived from corners and shots on target.
# 
# - data['pressure_index'] = (data['corners'] + data['shots_on_target']) / data['total_shots']
# 
# 5. Team Form: The recent win/loss record could be encoded as a feature, showing the momentum of the team.
# 
# - data['team_form'] = data['recent_matches'].apply(lambda x: calculate_form_from_recent_5_matches(x))
# 
# 6. Fatigue Factor: Taking into account the number of games played in a recent period to estimate player fatigue.
# 
# - data['fatigue_factor'] = data['matches_played_last_month'] / 30
# 

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define functions to fetch data from Bet365 API
def fetch_data(api_key, endpoint):
    url = f"https://api.b365api.com/v1/{endpoint}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def fetch_premier_league_matches(api_key, season):
    endpoint = f"soccer/matches?league=39&season={season}"
    return fetch_data(api_key, endpoint)

def fetch_fixture_statistics(api_key, fixture_id):
    endpoint = f"soccer/fixtures/statistics?fixture={fixture_id}"
    return fetch_data(api_key, endpoint)

# Calculate team form from recent 5 matches
def calculate_form_from_recent_5_matches(recent_matches):
    form_score = 0
    for match in recent_matches:
        if match == 'W':
            form_score += 3
        elif match == 'D':
            form_score += 1
    return form_score

# Fetch match data for multiple seasons and combine into a single DataFrame
api_key = 'xxxxx'
all_matches = []
for year in range(2010, 2023):
    season_data = fetch_premier_league_matches(api_key, year)
    if season_data:
        flattened_data = pd.json_normalize(season_data['data'])
        all_matches.extend(flattened_data.to_dict(orient='records'))

matches_df = pd.DataFrame(all_matches)

# Sort matches by date
matches_df.sort_values(by='fixture_date', inplace=True)
matches_df.reset_index(drop=True, inplace=True)

# Define the statistics to fetch for each fixture
statistic_types = [
    "Shots on Goal", "Shots off Goal", "Total Shots", "Blocked Shots",
    "Shots insidebox", "Shots outsidebox", "Fouls", "Corner Kicks",
    "Offsides", "Ball Possession", "Yellow Cards", "Red Cards",
    "Goalkeeper Saves", "Total passes", "Passes accurate", "Passes %",
    "Expected Goals"
]

# Create columns for each statistic for both home and away teams
modified_statistic_types = [
    f"home_{stat.replace(' ', '_').lower()}" for stat in statistic_types
] + [
    f"away_{stat.replace(' ', '_').lower()}" for stat in statistic_types
]

for stat_type in modified_statistic_types:
    matches_df[stat_type] = None

# Populate match statistics into the DataFrame
def populate_statistics(row):
    fixture_statistics = fetch_fixture_statistics(api_key, row['fixture_id'])
    if fixture_statistics and 'response' in fixture_statistics and len(fixture_statistics['response']) == 2:
        for stat in fixture_statistics['response'][0]['statistics']:
            column_name = 'home_' + stat['type'].lower().replace(' ', '_')
            row[column_name] = stat['value']
        for stat in fixture_statistics['response'][1]['statistics']:
            column_name = 'away_' + stat['type'].lower().replace(' ', '_')
            row[column_name] = stat['value']
    else:
        print(f"Data for fixture {row['fixture_id']} is incomplete or not available")
    return row

matches_df = matches_df.apply(populate_statistics, axis=1)

# Feature Engineering
matches_df['goal_conversion_rate'] = matches_df['goals_scored'] / matches_df['shots_on_target']
matches_df['defense_strength'] = matches_df['fouls_committed'] + (matches_df['yellow_cards'] * 2) + (matches_df['red_cards'] * 3)
matches_df['attacking_dominance'] = (matches_df['possession_percentage'] * matches_df['shots_on_target']) / 100
matches_df['pressure_index'] = (matches_df['corners'] + matches_df['shots_on_target']) / matches_df['total_shots']
matches_df['team_form'] = matches_df['recent_matches'].apply(lambda x: calculate_form_from_recent_5_matches(x))
matches_df['fatigue_factor'] = matches_df['matches_played_last_month'] / 30

# Save df
matches_df.to_csv('premier_league_matches_with_statistics.csv', index=False)

# Define features and target variable
features = ['goal_conversion_rate', 'defense_strength', 'attacking_dominance', 'pressure_index', 'team_form', 'fatigue_factor']
X = matches_df[features]
y = matches_df['match_result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {log_reg_accuracy * 100:.2f}%")

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

# Cross-validation for Logistic Regression
log_reg_cv_scores = cross_val_score(log_reg, X, y, cv=5)
print(f"Logistic Regression Cross-Validation Accuracy: {np.mean(log_reg_cv_scores) * 100:.2f}%")

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=5)
print(f"Random Forest Cross-Validation Accuracy: {np.mean(rf_cv_scores) * 100:.2f}%")
