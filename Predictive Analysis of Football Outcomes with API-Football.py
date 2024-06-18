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

# First we employ the requests library to obtain Premier League match data from API-Foootball.

# In[81]:


import requests
import pandas as pd
from pandas import json_normalize


# For the sake of modularity, we define two functions that fetch seasonal match data and fetch the specific match data, "fetch_premier_league_matches" and "fetch_fixture_statistics", respectively.
# 
# Terminology alert! In football, "Fixture" refers to the matches that has been played/yet to be played by a team.

# In[ ]:


def fetch_premier_league_matches(season):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
    querystring = {"league":"39", "season":str(season)}
    headers = {
        "X-RapidAPI-Key": "f6f92a0551msh8a9ff5ec6a49f33p1f8d7djsn8c257e0489d6",
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
def fetch_fixture_statistics(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v3/fixtures/statistics"
    querystring = {"fixture": str(fixture_id)}
    headers = {
        "X-RapidAPI-Key": "f6f92a0551msh8a9ff5ec6a49f33p1f8d7djsn8c257e0489d6",  
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code == 200:
        return response.json()
    else:
        return None


# In[120]:


all_matches = []
for year in range(2010, 2023):  
    season_data = fetch_premier_league_matches(year)
    flattened_data = json_normalize(season_data['response'], sep='_')
    all_matches.extend(flattened_data.to_dict(orient='records'))

matches_df = pd.DataFrame(all_matches)

matches_df.sort_values(by='fixture_date', inplace=True)

matches_df.reset_index(drop=True, inplace=True)

print(matches_df)


# In[114]:


matches_df.columns


# Now we look at each row in the 'matches_df' dataframe by the 'fixture_id', to call the function 'fetch_fixture_statistics' and add the information about the matches that we want into the 'matches_df'.

# In[117]:


statistic_types = [
    "Shots on Goal",
    "Shots off Goal",
    "Total Shots",
    "Blocked Shots",
    "Shots insidebox",
    "Shots outsidebox",
    "Fouls",
    "Corner Kicks",
    "Offsides",
    "Ball Possession",
    "Yellow Cards",
    "Red Cards",
    "Goalkeeper Saves",
    "Total passes",
    "Passes accurate",
    "Passes %",
    "Expected Goals"
]


# In[118]:


for stat_type in statistic_types:
    matches_df[stat_type] = None


# In[121]:


matches_df


# In[122]:


statistic_types = [
    "Shots on Goal",
    "Shots off Goal",
    "Total Shots",
    "Blocked Shots",
    "Shots insidebox",
    "Shots outsidebox",
    "Fouls",
    "Corner Kicks",
    "Offsides",
    "Ball Possession",
    "Yellow Cards",
    "Red Cards",
    "Goalkeeper Saves",
    "Total passes",
    "Passes accurate",
    "Passes %",
    "Expected Goals"
]

# new list with 'home_' and 'away_' prefixes
modified_statistic_types = [
    f"home_{stat.replace(' ', '_').lower()}" for stat in statistic_types
] + [
    f"away_{stat.replace(' ', '_').lower()}" for stat in statistic_types
]

print(modified_statistic_types)


# In[123]:


modified_statistic_types


# In[124]:


for stat_type in modified_statistic_types:
    matches_df[stat_type] = None


# In[125]:


matches_df


# In[ ]:


def populate_statistics(row):
    fixture_statistics = fetch_fixture_statistics(row['fixture_id'])

    if fixture_statistics and 'response' in fixture_statistics and len(fixture_statistics['response']) == 2:
        # Check if the response has two items (home and away team data)

        # Processing home team statistics
        for stat in fixture_statistics['response'][0]['statistics']:
            column_name = 'home_' + stat['type'].lower().replace(' ', '_')
            row[column_name] = stat['value']

        # Processing away team statistics
        for stat in fixture_statistics['response'][1]['statistics']:
            column_name = 'away_' + stat['type'].lower().replace(' ', '_')
            row[column_name] = stat['value']
    else:
        # Handle the case where fixture data is incomplete or not available
        print(f"Data for fixture {row['fixture_id']} is incomplete or not available")

    return row

# Apply the function to each row of matches_df
matches_df = matches_df.apply(populate_statistics, axis=1)

print(matches_df.head())

