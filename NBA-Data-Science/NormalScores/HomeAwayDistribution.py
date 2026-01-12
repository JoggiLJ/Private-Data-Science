import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Making sure the directory will work no matter who has downloads this NBA folder
file_dir = Path(__file__).resolve().parent.parent
csv_path = file_dir / 'Data' / 'Games.csv'

# Define the dataframe we want to use, use low_memory to avoid warnings about mixed types
Games = pd.read_csv(csv_path, low_memory= False)

# Make sure our data is in the proper format, coercing any deviant values
Games['gameDateTimeEst'] = pd.to_datetime(Games['gameDateTimeEst'], errors='coerce')

# Limit entries to 2024/2025 season (october 1st 2024 - july 1st 2025) 
# Make it a generic function, so analysis/visualization can be done in any time window/season
def limit_season(Games, startdate, enddate):
    Games_filtered = Games[(Games['gameDateTimeEst'] >= startdate) & (Games['gameDateTimeEst'] <= enddate)]
    return Games_filtered

# Call limiting function
Games = limit_season(Games, '2024-10-01', '2025-07-01')

# Make new df for relevant metrics. In this case only scores are interesting. They exist in homeScore and awayScore columns
Scores = Games[['awayScore', 'homeScore']]

#Computing means of home and away games
MuHome = Scores['homeScore'].mean()
MuAway = Scores['awayScore'].mean()

# Setting plot style, and defining colors as official NBA colors, for aesthetics
sns.set_style('whitegrid')
nba_blue ='#1D428A'
nba_red = '#C8102E'

#Plotting scores for home teams
sns.histplot(Scores['homeScore'],
             bins = range(0, 201, 5),
             color = nba_blue,
             kde = True,
             label = 'Score distribution of \nhome teams')

#Plotting scores for away teams
sns.histplot(Scores['awayScore'],
             bins = range(0, 201, 5),
             color = nba_red,
             kde = True,
             label = 'Score distribution of \naway teams')

#Visualizing the mean of home and away teams, and expressing the value down to 2 decimals
plt.axvline(MuHome, 
            color = nba_blue, 
            linestyle = '--',
            label = f'Mean score of home teams\n{MuHome:.2f}')
plt.axvline(MuAway, 
            color = nba_red, 
            linestyle = '--',
            label = f'Mean score of away teams\n{MuAway:.2f}')

#limiting the x axis and making the graph symmetric
plt.xlim(MuAway-60, MuHome+60)

#Adding legend and labels to axises
plt.legend(fontsize = 8)
plt.xlabel('Scores, in bins of 5')
plt.ylabel('Density')


plt.show()