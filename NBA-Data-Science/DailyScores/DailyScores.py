import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path
import sys

# Define the directories for calling the pull function,
# where to store/find the data, and where to find logos
ParentDir = Path(__file__).resolve().parent.parent
PullDir = ParentDir / 'PullData'
DataDir = ParentDir / 'Data' / 'Games.csv'
LogoDir = str(ParentDir / 'Data' / 'Logos')

# Define where the pull function is located
sys.path.insert(0, str(PullDir))

# Import the pull function
import PullDataNBA as pull # type: ignore

# Extract updatet data using the pull function.
# If kaggle doesn't work as intended, or the computer lacks internet, the rest of the analysis
# will be performed on old data already stored on the system
try:
    1/0 #TO PREVENT CONSTANT DOWNLOADS IN DEVELOPMENT. COMMENT WHEN SHIPPING!
    print('Downloading updated data from Kaggle...')
    df = pull.KaggleDownload('Games')

except:
    print('Warning: The program was not able to pull updated data.\nThe following analysis is based on old data that is locally stored')
    df = pd.read_csv(DataDir, low_memory = False)

# Define and drop irrelevant columns
DropCols = ['gameId', 'hometeamId', 'awayteamId', 'winner', 'attendance', 'arenaId', 'gameLabel', 'gameSubLabel', 'seriesGameNumber', 'gameType']
df = df.drop(columns = DropCols)

# Standardixe the dates and times in the gameDateTimeEst collumn
df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'], format='ISO8601', utc = True)

# Remove rhe time of day from all games, this information is not relevant
df['gameDateTimeEst'] =df['gameDateTimeEst'].dt.date

# Define the most recent day games were played
RecentDate = df['gameDateTimeEst'].max()

# Limit the data frame to the most recent date
df = df[df['gameDateTimeEst'] == RecentDate]

#Drop date column. No longer useful
df = df.drop(columns = 'gameDateTimeEst')

#Determine the number of games played on the most recent date
NumGames = len(df)
fig, axes = plt.subplots(NumGames, 1, figsize=(12, 2*NumGames))

if NumGames == 1:
    axes = [axes]

for i, game in df.iterrows():
    ax = axes[i]
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')

    AwayTeam = f'{game.iloc[0]}\n{game.iloc[1]}'
    AwayScore = game.iloc[4]
    AwayLogo = mpimg.imread(f'{LogoDir}/{game.iloc[1]}.png')

    HomeTeam = f'{game.iloc[2]}\n{game.iloc[3]}'
    HomeScore = game.iloc[5]   
    HomeLogo = mpimg.imread(f'{LogoDir}/{game.iloc[3]}.png')

    if HomeScore >= AwayScore:
        HomeWin = 'bold'
        AwayWin = 'light'
    else:
        HomeWin = 'light'
        AwayWin = 'bold'

    ax.text(0.5, 0.5, '-', fontsize = 20, ha = 'center', va = 'center', weight = 900)

    ax.text(0.25, 0.5, AwayTeam, fontsize = 16, ha = 'center',  va = 'center')
    ax.text(0.40, 0.5, AwayScore, fontsize = 20, ha = 'center', va = 'center', 
            weight = AwayWin)
    LogoAxAway = ax.inset_axes([0.05, 0.25, 0.15, 0.5])
    LogoAxAway.imshow(AwayLogo, aspect = 'equal')
    LogoAxAway.axis('off')

    ax.text(0.75, 0.5, HomeTeam, fontsize = 16, ha = 'center',  va = 'center')
    ax.text(0.60, 0.5, HomeScore, fontsize = 20, ha = 'center', va = 'center', 
            weight = HomeWin)
    LogoAxHome = ax.inset_axes([0.95, 0.25, 0.15, 0.5])
    LogoAxHome.imshow(HomeLogo, aspect = 'equal')
    LogoAxHome.axis('off')

plt.suptitle(f'NBA Scores for {RecentDate}', fontsize = 40, weight = 'bold')

plt.show()
