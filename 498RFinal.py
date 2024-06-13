# Final project for University of Maryland course MATH498R 
# Selected Topics in Mathematics; Experiential Learning: Mathematics of Sports Performance Analytics
# Spring 2024 tought by Professor Yanir Rubinstein
# By Arthur Lin and Jason Lott

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playbyplayv2
from nba_api.stats.endpoints import boxscoreadvancedv2
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.endpoints import gamerotation
from nba_api.stats.endpoints import boxscoreadvancedv3
# Imports the NBA API endpoint boxscoreadvancedv3 into the program. It is used to get the advanced stats for every player who played in a specified game.
from nba_api.stats.endpoints import boxscoretraditionalv3
# Imports the NBA API endpoint boxscoretraditionalv3 into the program. It is used to get the traditional stats for every player who played in a specified game.
from nba_api.stats.endpoints import TeamGameLogs
# Imports the NBA API endpoint TeamGameLogs into the program. It is used to gather game IDs for a specified season and/or team.
from sklearn.linear_model import LinearRegression
# Imports the Sci-Kit Learn model Linear Regression into the program. It is used to approximate the formula for some advanced stats.
from sklearn.ensemble import RandomForestRegressor
# Imports the Sci-Kit Learn model Random Forest Regressor into the program. It is used to approximate the formula for some advanced stats.
from sklearn.model_selection import train_test_split
# Imports the Sci-Kit Learn function train_test_split into the program. It is used to randomly split the data into training and testing datasets.
from tqdm import tqdm
# Imports the TQDM function tqdm. It is used to create progress bars to track progress of long data pulls.
import time
# Imports the time package into the program. It is used to delay the program in the retry wrapper.
import requests
# Imports the requests package into the program. It is used to handle exceptions in the retry wrapper.

# Get all players
all_players = players.get_players()
def get_player_id(player_name):
    # Iterate through the list of players
    for player in all_players:
        # Check if the player's name matches the input
        if player['full_name'].lower() == player_name.lower():
            # Return the player ID if found
            return player['id']
    # Return None if the player is not found
    return None

# Gather user input for a player's name
player_name = input("Enter a player's full name: ")
last_name = player_name.split(' ')[1]
player_id = get_player_id(player_name)
if player_id:
    print(f"The player ID for {player_name} is {player_id}.")
else:
    print(f"Player '{player_name}' not found.")

season = input('Enter the years a season spans (e.g. 2023-24): ')
gamefinder = leaguegamefinder.LeagueGameFinder(player_id_nullable = player_id, season_nullable= season)
games = gamefinder.get_data_frames()[0]
pd.set_option('display.max_columns',250)
pd.set_option('display.max_colwidth', None)
games

# Function to get game_id from game_date
def get_game_id(game_date):
    # Filter the DataFrame for the given game_date
    game = games[games['GAME_DATE'] == game_date]

    if len(game) > 0:
      return game['GAME_ID'].iloc[0]
    else:
      return "No game found for the given date."

# User input
game_date = input("Enter the game date (YYYY-MM-DD): ")
game_id = get_game_id(game_date)

# Get play-by-play data for the specified game
play_by_play = playbyplayv2.PlayByPlayV2(game_id=game_id)

# Set display for DataFrames
pd.set_option('display.max_columns',250)
pd.set_option('display.max_colwidth', None)

# Get play-by-play dataframe
pbp_df = play_by_play.get_data_frames()[0]

cleaned_pbp_df = pbp_df[['GAME_ID', 'EVENTMSGTYPE', 'PERIOD', 'PCTIMESTRING', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION', 'SCORE', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ABBREVIATION', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ABBREVIATION']]
pd.set_option('display.max_rows', None)
cleaned_pbp_df

home_boolean = True
game = games[games['GAME_DATE'] == game_date]
if game['MATCHUP'].str.contains('vs.').any():
  column = 'HOMEDESCRIPTION'
else:
  home_boolean = False
  column = 'VISITORDESCRIPTION'

column

# Function to convert PCTIMESTRING to total minutes elapsed
def convert_to_seconds(row):
    period = row['PERIOD']
    pctimestring = row['PCTIMESTRING']
    # Calculate the total seconds elapsed in the game
    seconds_elapsed = (int(period) - 1) * 12 * 60 + (12 * 60 - int(pctimestring.split(':')[0]) * 60) - int(pctimestring.split(':')[1])
    return seconds_elapsed

# Apply convert_to_seconds function to create a new column 'SECONDS_ELAPSED'
cleaned_pbp_df['SECONDS_ELAPSED'] = cleaned_pbp_df.apply(convert_to_seconds, axis=1)

# Separates the input period and time left
def parse_input(input_str):
    if not input_str:
        return None, None
    period, time_left = input_str.split(',')
    return int(period), time_left.strip()

def filter_time_frame(df, start_period=None, start_time=None, end_period=None, end_time=None):
    if start_period is not None and start_time is not None:
        start_seconds = (start_period - 1) * 12 * 60 + (12 * 60 - int(start_time.split(':')[0]) * 60) - int(start_time.split(':')[1])
        df = df[df['SECONDS_ELAPSED'] >= start_seconds]
    if end_period is not None and end_time is not None:
        end_seconds = (end_period - 1) * 12 * 60 + (12 * 60 - int(end_time.split(':')[0]) * 60) - int(end_time.split(':')[1])
        df = df[df['SECONDS_ELAPSED'] <= end_seconds]
    return df

time_frame_start = input("Enter a starting period and time left (e.g. 1, 11:45), or press enter to start at the beginning of the game: ")
time_frame_end = input("Enter the ending period and time left, or press enter to end at the end of the game: ")

start_period, start_time = parse_input(time_frame_start)
end_period, end_time = parse_input(time_frame_end)

cleaned_pbp_df = filter_time_frame(cleaned_pbp_df, start_period, start_time, end_period, end_time)

cleaned_pbp_df

"""###**Create DataFrame with all traditional and advanced stats**"""

# Use cleaned_pbp_df to get overall game stats for PIE calculation

# Game PTS
cleaned_pbp_df['SCORE'].iloc[0] = '0 - 0' # initialize starting score
cleaned_pbp_df['SCORE'].fillna(method='ffill', inplace=True)

# --- changed
cleaned_pbp_df['GAME_PTS'] = cleaned_pbp_df['SCORE'].apply(lambda x: np.sum(list(map(int, x.split(' - ')))))
# ---

# Game FGM
cleaned_pbp_df['GAME_FGM'] = (cleaned_pbp_df['EVENTMSGTYPE'] == 1).cumsum()

# Game FTM
cleaned_pbp_df['GAME_FTM'] = ((cleaned_pbp_df['EVENTMSGTYPE'] == 3) &
                               (~cleaned_pbp_df['HOMEDESCRIPTION'].str.contains('MISS', na=False)) &
                               (~cleaned_pbp_df['VISITORDESCRIPTION'].str.contains('MISS', na=False))).cumsum()

# Game FGA
cleaned_pbp_df['GAME_FGA'] = (cleaned_pbp_df['EVENTMSGTYPE'].isin([1, 2])).cumsum()

# Game FTA
cleaned_pbp_df['GAME_FTA'] = (cleaned_pbp_df['EVENTMSGTYPE'] == 3).cumsum()

# Game REB
cleaned_pbp_df['GAME_REB'] = (cleaned_pbp_df['EVENTMSGTYPE'] == 4).cumsum()

# Game AST
cleaned_pbp_df['GAME_AST'] = ((cleaned_pbp_df['HOMEDESCRIPTION'].str.contains('AST', na=False)) |
                                (cleaned_pbp_df['VISITORDESCRIPTION'].str.contains('AST', na=False))).cumsum()

# Game STL
cleaned_pbp_df['GAME_STL'] = ((cleaned_pbp_df['HOMEDESCRIPTION'].str.contains('STL', na=False)) |
                                (cleaned_pbp_df['VISITORDESCRIPTION'].str.contains('STL', na=False))).cumsum()
# Game BLK
cleaned_pbp_df['GAME_BLK'] = ((cleaned_pbp_df['HOMEDESCRIPTION'].str.contains('BLK', na=False)) |
                                (cleaned_pbp_df['VISITORDESCRIPTION'].str.contains('BLK', na=False))).cumsum()
# Game PF
cleaned_pbp_df['GAME_PF'] = (cleaned_pbp_df['EVENTMSGTYPE'] == 6).cumsum()

# Game TO
cleaned_pbp_df['GAME_TO'] = (cleaned_pbp_df['EVENTMSGTYPE'] == 5).cumsum()

cleaned_pbp_df

# Get team-specific stats needed for PER
cleaned_pbp_df['TEAM_AST'] = (cleaned_pbp_df[column].str.contains('AST', na=False)).cumsum()

team_abbreviation = games.loc[0, 'TEAM_ABBREVIATION']

team_scoring = cleaned_pbp_df[(cleaned_pbp_df['PLAYER1_TEAM_ABBREVIATION'] == team_abbreviation) &
                              (cleaned_pbp_df['EVENTMSGTYPE'] == 1)]

cleaned_pbp_df['TEAM_FGM'] = team_scoring['EVENTMSGTYPE'].cumsum()

cleaned_pbp_df

player_pbp = cleaned_pbp_df[cleaned_pbp_df['PLAYER1_ID'] == player_id]

# Add rows for substitution into the game and ends of quarters
# In EVENTMSGTYPE, 8 indicates substitutions
substitution_events = cleaned_pbp_df[(cleaned_pbp_df['EVENTMSGTYPE'] == 8) & (cleaned_pbp_df[column].str.contains(last_name))]

# In EVENTMSGTYPE, 13 indicates end of quarters
quarter_end = cleaned_pbp_df[cleaned_pbp_df['EVENTMSGTYPE'] == 13]

# Concatenate the DataFrames to player_pbp
player_pbp = pd.concat([player_pbp, substitution_events, quarter_end])

# Sort events sequentially by game time
player_pbp.sort_index(inplace=True)
player_pbp

# Plays where the player was categorized as PLAYER2 (we'll use to count steals, blocks, and assists)
second_pbp = cleaned_pbp_df[cleaned_pbp_df['PLAYER2_ID'] == player_id]
second_pbp

assists = second_pbp[second_pbp[column].str.contains(rf'{last_name} \d+ AST').fillna(False)]
player_pbp = pd.concat([player_pbp, assists], axis=0)

steals = second_pbp[second_pbp[column].str.contains(rf'\d+ STL').fillna(False)]
player_pbp = pd.concat([player_pbp, steals], axis=0)

blocks = second_pbp[second_pbp[column].str.contains(rf'{last_name} \d+ BLK').fillna(False)]
player_pbp = pd.concat([player_pbp, blocks], axis=0)
player_pbp.drop_duplicates(inplace = True)
player_pbp.sort_index()

# Traditional stats

# PTS
player_pbp['PTS'] = player_pbp.loc[player_pbp['PLAYER1_ID'] == player_id, column].str.extract(r'(\d+) PTS')

# AST
player_pbp['AST'] = player_pbp.loc[player_pbp['PLAYER2_ID'] == player_id, column].str.extract(r'(\d+) AST')

# OREB
player_pbp['OREB'] = player_pbp[column].str.extract(r'REBOUND \(Off:(\d+)')

# DREB
player_pbp['DREB'] = player_pbp[column].str.extract(r'REBOUND \(Off:\d+ Def:(\d+)')

# REB
player_pbp['REB'] = pd.to_numeric(player_pbp['OREB']) + pd.to_numeric(player_pbp['DREB'])

# STL
player_pbp['STL'] = player_pbp.loc[player_pbp['PLAYER2_ID'] == player_id, column].str.extract(r'(\d+) STL')

# BLK
player_pbp['BLK'] = player_pbp.loc[player_pbp['PLAYER2_ID'] == player_id, column].str.extract(r'(\d+) BLK')

# TO
player_pbp['TO'] = player_pbp[column].str.extract(r'Turnover \(P(\d+)\.T\d+\)')

# FOULS (committed)
player_pbp['PF'] = ((player_pbp['EVENTMSGTYPE'] == 6) & (player_pbp['PLAYER1_ID'] == player_id)).cumsum()

player_pbp.sort_index(inplace = True)
player_pbp.drop_duplicates(inplace=True)
player_pbp

# Shooting percentages

# FG
player_fg = cleaned_pbp_df[(cleaned_pbp_df['EVENTMSGTYPE'].isin([1, 2])) & (cleaned_pbp_df['PLAYER1_ID'] == player_id)]
player_fg["FGM"] = (player_fg["EVENTMSGTYPE"] == 1).cumsum()
player_fg["FGA"] = player_fg.reset_index(drop = True).index + 1
player_fg["FG%"] = (player_fg["FGM"] / player_fg["FGA"] * 100).round(2)
player_pbp = pd.concat([player_pbp, player_fg], axis=0)
player_pbp.update(player_fg[['FGM', 'FGA', 'FG%']])

#	FG3

# Create condition for removing rows
mask = player_fg[column].str.contains('3PT')

# Find rows indicating 3-pointers
player_3pt_attempts = player_fg[mask]

player_3pt_attempts["3PTM"] = (player_3pt_attempts["EVENTMSGTYPE"] == 1).cumsum()
player_3pt_attempts["3PTA"] = player_3pt_attempts.reset_index(drop = True).index + 1
player_3pt_attempts["3PT%"] = (player_3pt_attempts["3PTM"] / player_3pt_attempts["3PTA"] * 100).round(2)
player_pbp = pd.concat([player_pbp, player_3pt_attempts], axis=0)
player_pbp.update(player_3pt_attempts[['3PTM', '3PTA', '3PT%']])

# FT
player_free_throws = cleaned_pbp_df[(cleaned_pbp_df['EVENTMSGTYPE'] == 3) & (cleaned_pbp_df['PLAYER1_ID'] == player_id)]
total_made = 0
total_attempts = 0

ft_made = []
ft_attempted = []

for index, row in player_free_throws.iterrows():
    # Check if the free throw attempt was made
    if 'MISS' not in row[column]:
      total_made += 1

    ft_made.append(total_made)
    total_attempts += 1
    ft_attempted.append(total_attempts)

player_free_throws['FTM'] = ft_made
player_free_throws['FTA'] = ft_attempted
player_free_throws['FT%'] = (player_free_throws['FTM'] / player_free_throws['FTA'] * 100).round(2)
player_pbp = pd.concat([player_pbp, player_free_throws], axis=0)
player_pbp.update(player_free_throws[['FTM', 'FTA', 'FT%']])

# Dropping duplicate rows
# Reset the index to make the current index a column
player_pbp.reset_index(inplace=True)

# Drop duplicate rows based on the index column
player_pbp.drop_duplicates(subset='index', inplace=True)

# Drop the additional index column created by reset_index
player_pbp.drop(columns=['index'], inplace=True)

player_pbp

# Minutes played and game pace for TENDEX calculation

# MINUTES PLAYED

# Get complete list of substitutions for the target player's team
subs_list = gamerotation.GameRotation(game_id= game_id)
subs_list_df = pd.DataFrame()
if home_boolean:
  subs_list_df = subs_list.data_sets[1].get_data_frame()
else:
  subs_list_df = subs_list.data_sets[0].get_data_frame()
subs_list_df = subs_list_df[subs_list_df["PERSON_ID"] == player_id]
subs_list_df["IN_TIME_REAL"] = subs_list_df["IN_TIME_REAL"] / 10
subs_list_df["OUT_TIME_REAL"] = subs_list_df["OUT_TIME_REAL"] / 10
subs_list_df

# new code ---
bench_tuples = []
for i in range(len(subs_list_df)):
  curr_row = subs_list_df.iloc[i]
  if i == 0 and curr_row["IN_TIME_REAL"] != 0.0:
    bench_tuples.append((0.0, curr_row["IN_TIME_REAL"]))

  if i != len(subs_list_df)-1:
    next_row = subs_list_df.iloc[i+1]
    bench_tuples.append((curr_row["OUT_TIME_REAL"],next_row["IN_TIME_REAL"]))
  elif curr_row["OUT_TIME_REAL"] != 2880.0:
    bench_tuples.append((curr_row["OUT_TIME_REAL"], 2880.0))

# ---

# Initialize the ON_COURT column with zeros
player_pbp['ON_COURT'] = 0
# Initialize the MIN_PLAYED column
player_pbp['MIN_PLAYED'] = player_pbp['SECONDS_ELAPSED']

# Iterate over each row in player_pbp dataframe
for index, row in player_pbp.iterrows():

    # Get the current time in seconds
    current_time = row['SECONDS_ELAPSED']

    # Check if the player was on the court at this time
    if any((subs_list_df['IN_TIME_REAL'] <= current_time) & (subs_list_df['OUT_TIME_REAL'] > current_time)):
        player_pbp.at[index, 'ON_COURT'] = 1

    # Update MIN_PLAYED column for subsequent rows where the player is on the court
    if player_pbp.at[index, 'ON_COURT'] == 1 and index > 0:
        prev_time = player_pbp.at[index - 1, 'SECONDS_ELAPSED']
        player_pbp.at[index, 'MIN_PLAYED'] = player_pbp.at[index - 1, 'MIN_PLAYED'] + (current_time - prev_time)
    elif index > 0:
        # If player is not on the court, keep MIN_PLAYED value the same as the previous row
        player_pbp.at[index, 'MIN_PLAYED'] = player_pbp.at[index - 1, 'MIN_PLAYED']

# Convert ON_COURT column to boolean
player_pbp['ON_COURT'] = player_pbp['ON_COURT'].astype(bool)

# Convert MIN_PLAYED column to minutes
player_pbp['MIN_PLAYED'] = (player_pbp['MIN_PLAYED'] / 60).round(3)

# -----
# PACE

# Get advanced box score data for the specified game
box_advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)

# Get advanced stats for the player
player_box_advanced_df = box_advanced.get_data_frames()[1]
if home_boolean:
  advanced_stat_index = 0
else:
   advanced_stat_index =1

player_box_advanced_df = player_box_advanced_df[player_box_advanced_df['TEAM_ABBREVIATION'] == player_pbp['PLAYER1_TEAM_ABBREVIATION'].iloc[advanced_stat_index]]

# Assign pace to a new column 'PACE' in player_pbp
player_pbp['PACE'] = player_box_advanced_df['PACE'].iloc[0]

# Set stats that are NaN to 0 at the start of the DataFrame
player_pbp.iloc[0] = player_pbp.iloc[0].fillna(0)
stat_columns = ['TEAM_FGM', 'TEAM_AST', 'PACE', 'MIN_PLAYED', 'PTS', 'AST', 'OREB', 'DREB', 'REB', 'STL', 'BLK', 'TO', 'PF', 'FGM', 'FGA', 'FG%', '3PTA', '3PTM', '3PT%', 'FTA', 'FTM', 'FT%']
player_pbp[stat_columns] = player_pbp[stat_columns].ffill()

# Convert columns to numeric
player_pbp[stat_columns] = player_pbp[stat_columns].apply(pd.to_numeric, errors='coerce')

player_pbp

season_game_finder = leaguegamefinder.LeagueGameFinder(season_nullable= season, league_id_nullable= '00')
season_games = season_game_finder.get_data_frames()[0]
season_league_averages = season_games[['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'REB', 'AST', 'TOV', 'PF']].mean()
season_league_averages = pd.DataFrame(season_league_averages)
season_league_averages = season_league_averages.T
season_league_averages.apply(pd.to_numeric)

# Advanced stats

# Assign stats to variables
PTS = player_pbp['PTS']
AST = player_pbp['AST']
OREB = player_pbp['OREB']
DREB = player_pbp['DREB']
REB = player_pbp['REB']
STL = player_pbp['STL']
BLK = player_pbp['BLK']
TO = player_pbp['TO']
PF = player_pbp['PF']
FGM = player_pbp['FGM']
FGA = player_pbp['FGA']
FTM = player_pbp['FTM']
FTA = player_pbp['FTA']
FG3M = player_pbp['3PTM']
MIN = player_pbp['MIN_PLAYED']
PACE = player_pbp['PACE']
GAME_PTS = player_pbp['GAME_PTS']
GAME_AST = player_pbp['GAME_AST']
GAME_REB = player_pbp['GAME_REB']
GAME_STL = player_pbp['GAME_STL']
GAME_BLK = player_pbp['GAME_BLK']
GAME_TO = player_pbp['GAME_TO']
GAME_PF = player_pbp['GAME_PF']
GAME_FGM = player_pbp['GAME_FGM']
GAME_FGA = player_pbp['GAME_FGA']
GAME_FTM = player_pbp['GAME_FTM']
GAME_FTA = player_pbp['GAME_FTA']

# ----------
# eFG%

# Calculate eFG% and handle division by zero
eFG = []
for index, row in player_pbp.iterrows():
    if row['FGA'] == 0:
        eFG.append(0)
    else:
        eFG.append((row['FGM'] + 0.5 * row['3PTM']) / row['FGA'])

player_pbp['eFG%'] = [round(efg * 100, 2) for efg in eFG]
# ----------
# true shooting

# Convert PTS to numeric for calculation
player_pbp['PTS'] = player_pbp['PTS'].apply(pd.to_numeric, errors='coerce')

# Calculate TS% and handle division by zero
TS_percentages = []
for index, row in player_pbp.iterrows():
    if row['FGA'] == 0 and row['FTA'] == 0:
        TS_percentages.append(0)
    else:
        TS_percentages.append(row['PTS'] / (2 * (row['FGA'] + 0.44 * row['FTA'])))

player_pbp['TS%'] = [round(ts * 100, 2) for ts in TS_percentages]
# ----------
# PER

# Calculating all variables needed for PER

# factor, VOP, and DRBP for unadjusted PER calculation
lgPTS = season_league_averages['PTS'].iloc[0]
lgFGM = season_league_averages['FGM'].iloc[0]
lgFGA = season_league_averages['FGA'].iloc[0]
lgFG3M = season_league_averages['FG3M'].iloc[0]
lgFG3A = season_league_averages['FG3A'].iloc[0]
lgFTM = season_league_averages['FTM'].iloc[0]
lgFTA = season_league_averages['FTA'].iloc[0]
lgOREB = season_league_averages['OREB'].iloc[0]
lgREB = season_league_averages['REB'].iloc[0]
lgAST = season_league_averages['AST'].iloc[0]
lgTO = season_league_averages['TOV'].iloc[0]
lgPF = season_league_averages['PF'].iloc[0]

tmAST = player_pbp['TEAM_AST']
tmFGM = player_pbp['TEAM_FGM']

# factor
factor = (2/3) - ((.5 * (lgAST / lgFGM)) / (2 * (lgFGM / lgFTM)))

# VOP
VOP = lgPTS / (lgFGA - lgOREB + lgTO + 0.44 * lgFTA)

# DRBP
DRBP = (lgREB - lgOREB) / lgREB

# PER
player_pbp['PER'] = (100 / MIN) * (FG3M - ((PF * lgFTM) / lgPF) + ((FTM / 2) * (2 - (tmAST/(3 * tmFGM)))) + (FGM * (2 - ((factor * tmAST) / tmFGM))) + ((2 * AST) / 3) + VOP * (DRBP * (2 * OREB + BLK - 0.2464 * (FTA - FTM) - (FGA - FGM) - REB) + ((0.44 * lgFTA * PF) / lgPF) - TO - OREB + STL + REB - 0.1936 * (FTA - FTM)))

# ----------
# PIE
player_pbp['PIE'] = 100 * (PTS + FGM + FTM - FGA - FTA + DREB + OREB/2 + AST + STL + BLK/2 - PF - TO) / (GAME_PTS + GAME_FGM + GAME_FTM - GAME_FGA - GAME_FTA + (1.5 * GAME_REB) + GAME_AST + GAME_STL + GAME_BLK/2 - GAME_PF - GAME_TO)

# ----------
# TENDEX

player_pbp['TENDEX'] = (PTS + OREB + DREB + AST + STL + BLK - (FGA - FGM) - (0.5 * (FTA - FTM)) - TO - PF) / (MIN * PACE)

player_pbp.iloc[0] = player_pbp.iloc[0].fillna(0)
player_pbp.ffill(inplace = True)
player_pbp

"""###**a) plot a "flow graph" for a given stat for a given player in a given game or time frame in the game**"""

# Filter player_pbp dataframe based on ON_COURT column
player_pbp_on_court = player_pbp[player_pbp['ON_COURT']]
player_pbp_off_court = player_pbp[~player_pbp['ON_COURT']]

# Customize x-axis ticks
max_seconds = player_pbp['SECONDS_ELAPSED'].max()
num_quarters = max_seconds // (12 * 60)
ticks = np.arange(0, max_seconds + 1, 12 * 60)  # Beginning of game and ends of quarters
plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Let user input a basic stat
basic_stat = input('Enter a basic stat (PTS, AST, OREB, DREB, REB, STL, BLK, TO, PF) to create a flow graph of: ')

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court[basic_stat]
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(0, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel(f' {basic_stat}')
plt.title(f' Progression of {basic_stat} for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""###**b) same as a) but for also for stats like FT% and FG% and 3FG% again as a function of time**"""

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['FT%']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(0, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('Free Throw Percentage')
plt.title(f'Free Throw Percentage Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['FG%']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(0, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('Field Goal Percentage')
plt.title(f'Field Goal Percentage Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['3PT%']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(0, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('Three Point Percentage')
plt.title(f'Three Point Percentage Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""###**c) similar but different "flow graph" for, say, FG% for given player or team as a function of the game in the season (1 to 82). Here there are two versions to consider:**

c.1) FG% (say) for each individual game;

c.2) FG% (say) cumulative for shots up until that game in the season. This would clearly plateau as we approach game 82.
"""

# Make percentages whole numbers (for aesthetic purposes on plot)
games[['FG_PCT', 'FG3_PCT', 'FT_PCT']] *= 100
games.fillna(0, inplace = True)
games

# Plotting FG% for each individual game of the season
x = games.index
y = games['FG_PCT']
plt.plot(x, y)
plt.xlabel('Games Played')
plt.ylabel('FG%')
plt.title(f'FG% by Game for {player_name} in the {season} season')

plt.xlim(x.min(), x.max())
plt.ylim(0, y.max() + 5)
plt.grid(True)
plt.show()

games['CUMULATIVE_FG_PCT'] = (games['FGM'].cumsum() / games['FGA'].cumsum() * 100).round(2)
games['CUMULATIVE_FG3_PCT'] = (games['FG3M'].cumsum() / games['FG3A'].cumsum() * 100).round(2)
games['CUMULATIVE_FT_PCT'] = (games['FTM'].cumsum() / games['FTA'].cumsum() * 100).round(2)
games

# Plotting cumulative FG% over the course of the season
x = games.index
y = games['CUMULATIVE_FG_PCT']
plt.plot(x, y)
plt.xlabel('Games Played')
plt.ylabel('Cumulative FG%')
plt.title(f'Cumulative FG% for {player_name} in the {season} season')

plt.xlim(x.min(), x.max())
plt.ylim(0, y.max() + 5)
plt.grid(True)
plt.show()

"""###**d) same as a) but for the advanced stats BPM, eFG, true shooting, RPM, PER, VORP, TENDEX. Include discussions of the formulas for each and their derivation. See, e.g., https://en.wikipedia.org/wiki/Advanced_statistics_in_basketball**

**eFG%** (effective field goal percentage) adjusts FG% to account for the point difference between three-pointers and all other field goals. It shows the FG% that a player shooting only 2-pointers would have to achieve to match output of a player shooting 2s and 3s.

*Using any of these formulas, eFG% for a good 3PT shooter can be over 100% (not realistic)

*   (FGM + 0.5(FG3M)) / FGA
*   (PPG - FTM) / 2(FGA)
  *   Highlights the logic behind eFG% by pretending that a player only shoots 2-pointers
*   (FG2M + 1.5(FG3M)) / FGA
"""

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['eFG%']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(0, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('eFG%')
plt.title(f'Effective FG% Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""**True Shooting** measures a player's efficiency at shooting the ball. All field goals and free throws are included in its calculation.

*   PTS / 2(FGA + (0.44(FTA)))

"""

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['TS%']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(0, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('TS%')
plt.title(f'True Shooting % Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""**PIE** (player impact estimate) gauges a player's overall contribution to a game. It yields comparable results to PER, with a much simpler calculation and more reliable measure of a player's defensive impact.


(PTS + FGM + FTM - FGA - FTA + DREB + (.5 * OREB) + AST + STL + (.5 * BLK) - PF - TO) / (GmPTS + GmFGM + GmFTM - GmFGA - GmFTA + GmDREB + (.5 * GmOREB) + GmAST + GmSTL + (.5 * GmBLK) - GmPF - GmTO)
"""

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['PIE']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
y = player_pbp['PIE']
plt.xlim(x.min(), x.max())
plt.ylim(y.min() - 20, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('PIE')
plt.title(f'Player Impact Estimate Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""**PER** (player efficiency rating) measures a player's per-minute performance, while adjusting for pace.

*   35+: all-time great season
*   15-16.5: slightly above-average player
*   0-9: player who won't stick in the league






"""

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['PER']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
plt.xlim(x.min(), x.max())
plt.ylim(y.min() - 5, y_on_court.max() + 2)
plt.xlabel('Game Time')
plt.ylabel('PER')
plt.title(f'Player Efficiency Rating Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""**TENDEX** is the original weighted advanced stat and measures a player's overall efficiency.

(PTS) + (REB) + (AST) + (STL) + (BLK) - (Missed FG) - 0.5 * (Missed FT) - (TO) - (Fouls Committed) / (MIN played) / (PACE)
"""

player_pbp_off_court

player_pbp_on_court

plt.xticks(ticks, ['Q1 Start'] + [f'End Q{i}' for i in range(1, int(num_quarters) + 1)])

# Plotting while on court
x_on_court = player_pbp_on_court['SECONDS_ELAPSED']
y_on_court = player_pbp_on_court['TENDEX']
plt.plot(x_on_court, y_on_court, linestyle='-', label='On Court')

# Plotting while off court
# New code ---
label = 'On Bench'
for e in bench_tuples:
  plt.plot([e[0], e[1]], [np.interp(e[0], x_on_court, y_on_court), np.interp(e[1], x_on_court, y_on_court)], linestyle='-', color='red', label= label)
  label = "_nolegend_"
# ---

x = player_pbp['SECONDS_ELAPSED']
y = player_pbp['TENDEX']
plt.xlim(x.min(), x.max())
plt.ylim(y.min() - 0.002, y.max() + 0.002)
plt.xlabel('Game Time')
plt.ylabel('Tendex')
plt.title(f'Tendex Over the Game for {player_name}')

plt.legend()
plt.grid(True)
plt.show()

"""**BPM** (box plus/minus) estimates a player's contribution to the team when they're on the court. A player’s box score information, position, and the team’s overall performance are used to estimate the player’s contribution in points above league average per 100 possessions played. BPM does not take into account playing time.

**VORP** (value over replacement player) converts the BPM rate into an estimate of each player's overall contribution to the team, measured vs. what a theoretical "replacement player" would provide, where the "replacement player" is defined as a player on minimum salary or not a normal member of a team's rotation.

###**e) given an advanced stat discover its formula using regression techniques.**

e.1) First, as a practice case, take one of the stats from d) for which we have the formula. Suppose the formula depends on 5 box-score data points (e.g., rebounds, points, assists, turnovers, steals, blocks). Determine, solely from data the approximate parameters for the formula.
"""

def retry(func, retries=10):
    def retry_wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(e)
            time.sleep(np.random.normal(2, 0.25, 1)[0])
            attempts += 1
    return retry_wrapper
# Our retry wrapper sleeps a random amount of time based on a normal distribution with mean 2 and standard deviation 0.25 when an API call results in an exception. The function will retry a call up to 10 times.

@retry
def get_full_season_game_ids(season_str):
  single_season_log = TeamGameLogs(season_nullable=season_str, season_type_nullable="Regular Season", league_id_nullable="00")
  single_season_logs = single_season_log.get_data_frames()[0]
  return np.unique(single_season_logs["GAME_ID"].to_list())
# This function get_full_season_game_ids takes in a string in the form "YYYY-YY" which represents an NBA season. The function returns a list with the ID of every game played in that regular season by accessing the TeamGameLogs endpoint of the NBA API. Tagged with the retry annotation to handle kickback from the API.

@retry
def get_adv_BS(game):
  return boxscoreadvancedv3.BoxScoreAdvancedV3(game).data_sets[0].get_data_frame()
# This function get_adv_BS takes in an int which represents a game id. The function returns a Pandas data frame containing the advanced stats of every player who played in the game by accessing the BoxScoreAdvancedV3 endpoint of the NBA API. Tagged with the retry annotation to handle kickback from the API.

@retry
def get_trad_BS(game):
  return boxscoretraditionalv3.BoxScoreTraditionalV3(game).data_sets[0].get_data_frame()
# This function get_trad_BS takes in an int which represents a game id. The function returns a Pandas data frame containing the traditional stats of every player who played in the game by accessing the BoxScoreTraditionalV3 endpoint of the NBA API. Tagged with the retry annotation to handle kickback from the API.

game_ids = get_full_season_game_ids("2022-23")
# Call get_full_season_game_ids on the 2022-23 season and store the resulting list in new variable game_ids.

adv_BSs = []
# Create variable adv_BSs with default value of an empty list. This will be used to hold a list of all the different advanced stats.
for game in tqdm(game_ids):
# Start looping through every element in game_ids where the variable game stores the current game's ID.
  adv_BSs.append(get_adv_BS(game))
  # Call get_adv_BS with the current game's ID and appends the resulting data frame to the list adv_BSs.
adv_BS_df = pd.concat(adv_BSs)
# Using Pandas function concat, concatenate every data frame in list adv_BSs into one data frame with all advanced stats for every player in every game in the 2022-23 NBA regular season, stored in new variable adv_BS_df.
adv_BS_df.head()

adv_BS_df.to_csv("adv_BS_2022_23.csv")
# Using Pandas function to_csv, save the data frame adv_BS_df locally as a comma serperated values (CSV) file named adv_BS_2022_23.csv so it can be easily accessed in the future.

trad_BSs = []
for game in tqdm(game_ids):
  trad_BSs.append(get_trad_BS(game))
trad_BS_df = pd.concat(trad_BSs)
trad_BS_df.head()

trad_BS_df.to_csv("trad_BS_2022_23.csv")
# Repeat the process to gather all traditional stats for every player in every game from the 2022-23 NBA regular season and save the resulting data frame locally as trad_BS_2022_23.csv.

adv_BS_df = pd.read_csv("adv_BS_2022_23.csv")
trad_BS_df = pd.read_csv("trad_BS_2022_23.csv")
# Using Pandas function read_csv, read the advanced and traditional stats data frames back into the respective variables adv_BS_df and trad_BS_df.

full_BS_df = trad_BS_df.merge(adv_BS_df, on= ["gameId", "personId"], how= "inner")[["gameId", "personId", "nameI_x", "fieldGoalsAttempted", "freeThrowsAttempted", "points", "trueShootingPercentage"]]
# Using Pandas function merge, inner join the traditional and advanced stats data frames based on the game's ID and the player's ID. After joining the data frames, compress the data to only keep columns that are used in calculating true shooting percentage and store the resulting data frame in new variable full_BS_df.

full_BS_df.head()

X = full_BS_df[["fieldGoalsAttempted", "freeThrowsAttempted", "points"]]
# Get the stats true shooting percentage is calculated from and store them in new variable X for regression.
y = full_BS_df["trueShootingPercentage"]
# Get true shooting percentage and store it in new variable y for regression.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Using Sci-Kit Learn function train_test_split randomly split the data into train and test group where the training group contains 90% of the original data. They are stored in new variables X_train, X_test, y_train, and y_test.
lin_reg = LinearRegression()
# Create a Sci-Kit Learn LinearRegression model and store it in the new variable lin_reg.
lin_reg.fit(X_train, y_train)
# Using Sci-Kit Learn function fit, train the linear regression model on the designated training data.
print(f"Variable weights: {dict(zip(X.columns, lin_reg.coef_))}")
# Print out the variable's weights assigned by the linear regression model by creating a dictionary where the keys are the variables and the values are their weights from the model's field coef_.
print(f"Intercept: {lin_reg.intercept_}")
# Print out the model's intercept from its field intercept_.
print(f"R-squared score: {lin_reg.score(X_test, y_test)}")
# Using Sci-Kit Learn function score, print out the model's R^2 score on the designated test data.
print(f"Mean absolute error: {np.mean(abs(lin_reg.predict(X_test)- y_test))}")
# Using Sci-Kit Learn function predict, print out the manually calculated mean absolute error on the designated test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# Randomly split the original data into new train and test sets.
rf_reg = RandomForestRegressor()
# Create a Sci-Kit Learn RandomForestRegressor model and store it in the new variable rf_reg.
rf_reg.fit(X_train, y_train)
# Using Sci-Kit Learn function fit, train the random forest regressor model on the designated training data.
print(f"Variable importance: {dict(zip(X.columns, rf_reg.feature_importances_))}")
# Print out the feature's importances assigned by the random forest regressor model by creating a dictionary where the keys are the features and the values are their importances from the model's field feature_importances_.
print(f"R-squared score: {rf_reg.score(X_test, y_test)}")
# Using Sci-Kit Learn function score, print out the model's R^2 score on the designated test data.
print(f"Mean absolute error: {np.mean(abs(rf_reg.predict(X_test)- y_test))}")
# Using Sci-Kit Learn function predict print out the manually calculated mean absolute error on the designated test data.

"""e.2) Second, there are "black-box stats" out there for which the exact formula is not published. Determine an approximate formula for the PPA stat -
https://kevinbroom.com/ppa/
This stat also inputs possessions each player played.
"""

PPA_df = pd.read_csv("2022_23_PPA.csv")
# Read player production average (PPA) data from 2022_23_PPA.csv which was obtained from Kevin Broom's website. It contains the PPA value for every player whio played in an NBA game in the 2022-23 season before the all star break. Store the resulting data frame in new variable PPA_df.
PPA_df = PPA_df[["Player", "Tm", "PPA"]]
# Compress the PPA data frame to only the columns Player, Tm, and PPA.
PPA_df["player_id"] = PPA_df["Player"].apply(get_player_id)
# Using Pandas function apply with custom function get_player_id on the PPA data frame along the first axis, create new column player_id by finding the corresponding ID for every player.
PPA_df_safe = PPA_df.dropna(subset= ["player_id"])
# Player names with accents and  apostrophes in the PPA data from Kevin Broom's website did not match with the names in the NBA API. Therefore we were not able to match them to the correct player ID. We decided to remove these rows from the data frame and store the resulting data frame in new variable PPA_df_safe.
PPA_df_safe = PPA_df_safe.astype({"player_id": int})
# Cast the player ID column from type String to int.
PPA_df_safe[["avg_min", "avg_pts", "avg_fga", "avg_oreb", "avg_dreb", "avg_ast", "avg_stl", "avg_blk", "avg_tov", "avg_pf", "avg_dr", "starts"]] = np.nan
# Create new columns for the stats that are used in the calculation of PPA. Set the default value of all the columns to be empty, they will be filled in later.

single_season_logs = TeamGameLogs(season_nullable="2022-23", season_type_nullable='Regular Season', league_id_nullable='00').get_data_frames()[0]
# Using the NBA API endpoint TeamGameLogs, get information about every NBA regular season game from the 2022-23 season. Store the resulting data frame in new variable
before_AS_game_ids = np.unique(single_season_logs[(single_season_logs["GAME_DATE"] > "2022-10-17") & (single_season_logs["GAME_DATE"] < "2023-02-17")]["GAME_ID"])
# Filter the data frame to only include the game played before the all star break in the 2022-23 season and store the ID of every game in new variable before_AS_game_ids.
statlines = trad_BS_df.merge(adv_BS_df, on= ["gameId", "personId"], how= "inner")
# Using Pandas function merge, inner join the traditional and advanced stats data frames based on the game's ID and the player's ID. Store the resulting data frame in new variable statlines.
statlines = statlines[statlines["gameId"].isin(before_AS_game_ids.astype(int))]
# Filter the data frame to only include games before the all star break.
statlines.dropna(subset= ["minutes_x"], inplace= True)
# Remove all rows where the minutes stat is empty implying the player never checked into the game.
statlines.head()

def get_minutes(row):
  if type(row["minutes_x"]) == str:
    mins, secs = row["minutes_x"].split(":")
    mins = int(mins)
    secs = int(secs)
    return mins + secs/60
  return row["minutes_x"]
# Create function get_minutes that takes in a row of the stat lines data frame and converts the string representing minutes played into a float.

statlines["minutes_float"] = statlines.apply(get_minutes, axis= 1)
# Using Pandas function apply with custom function get_minutes on the stat lines data frame along the first axis, create new column minutes_float.

@retry
def get_game_rotation(game):
  dfs = gamerotation.GameRotation(game_id= game).get_data_frames()
  return pd.concat(dfs)

game_subs_dfs = []
for game in tqdm(before_AS_game_ids):
  game_subs_dfs.append(get_game_rotation(game))
game_subs_df = pd.concat(game_subs_dfs)
game_subs_df.to_csv("PPA_subs.csv")

game_subs_df = pd.read_csv("PPA_subs.csv")

def add_starts(row):
  game = row["gameId"]
  player = row["personId"]
  player_subs = game_subs_df[(game_subs_df["GAME_ID"] == f"00{game}") & (game_subs_df["PERSON_ID"] == player)]
  if len(player_subs) > 0 and player_subs.iloc[0]["IN_TIME_REAL"] == 0.0:
    return 1
  return 0

statlines["started?"] = statlines.apply(add_starts, axis= 1)

def get_stats(row):
  player = int(row["player_id"])
  team = row["Tm"]
  # Create function get_stats and variables player and team which represent the current row's player and team.
  filtered_statlines = statlines[(statlines["personId"] == player) & (statlines["teamTricode_x"] == team)]
  # Filter the stat lines data frame to only include the current row's player and team. This is necessary because if a player played on multiple teams they have separate PPA values. Store the resulting data frame in new variable filtered_statlines.
  avg_stats = filtered_statlines.agg({
      "minutes_float": "mean",
      "points": "mean",
      "fieldGoalsAttempted": "mean",
      "reboundsOffensive": "mean",
      "reboundsDefensive": "mean",
      "assists": "mean",
      "steals": "mean",
      "blocks": "mean",
      "turnovers": "mean",
      "foulsPersonal": "mean",
      "defensiveRating": "mean",
      "started?": "sum"
  })
  # Using Pandas function agg, aggregate the filtered stat lines to get the average minutes, points,field goal attempts, offensive and defensive rebounds, assists, steals, blocks, turnovers, personal fouls, and defensive rating. Store the resulting series in new variable avg_stats.
  row[["avg_min", "avg_pts", "avg_fga", "avg_oreb", "avg_dreb", "avg_ast", "avg_stl", "avg_blk", "avg_tov", "avg_pf", "avg_dr", "starts"]] = avg_stats.to_list()
  return row
  # Set the current row's average stats the result of the calculations in the last line. Return the updated version of the current row.

PPA_df_safe = PPA_df_safe.apply(get_stats, axis= 1)
# Using Pandas function apply with custom function get_stats on the PPA data frame along the first axis, gather every player's average stats.

PPA_df_safe.head()

PPA_df_safe.dropna(inplace= True)
# Remove any more rows that still have empty values.
X = PPA_df_safe[['avg_min', 'avg_pts', 'avg_fga', 'avg_oreb', 'avg_dreb', 'avg_ast', 'avg_stl', 'avg_blk', 'avg_tov', 'avg_pf', 'avg_dr', "starts"]]
# Get the stats PPA is calculated from and store them in new variable X for regression.
y = PPA_df_safe["PPA"]
# Get PPA and store it in new variable \textit{y} for regression.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(f"Varaible weights: {dict(zip(X.columns, lin_reg.coef_))}")
print(f"Intercept: {lin_reg.intercept_}")
print(f"R-squared score: {lin_reg.score(X_test, y_test)}")
print(f"Mean absolute error: {np.mean(abs((X_test @ lin_reg.coef_ + lin_reg.intercept_) - y_test))}")
# Randomly split the data into training and testing sets. Create and train Sci-Kit Learn linear regression model. Print model details and some summary statistics of the model. Identical code to true shooting percentage linear regression.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
rf_reg = RandomForestRegressor(n_estimators= 50)
rf_reg.fit(X_train, y_train)
print(f"Variable importance: {dict(zip(X.columns, rf_reg.feature_importances_))}")
print(f"R-squared score: {rf_reg.score(X_test, y_test)}")
print(f"Mean absolute error: {np.mean(abs(rf_reg.predict(X_test)- y_test))}")
# Randomly split the data into training and testing sets. Create and train Sci-Kit Learn random forest regressor model. Print model details and some summary statistics of the model. Identical code to true shooting percentage random forest regressor.