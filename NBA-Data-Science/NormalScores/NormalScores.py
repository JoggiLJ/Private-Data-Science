import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import scipy.stats as stats

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

# Create one big collumn with all scores from both home and away teams. This is easier to visualize
TotalScores = Scores['awayScore']
TotalScores = pd.concat([TotalScores, Scores['homeScore']])

# Calculating mean and standard deviation for the data, to be used later
MuTotal, SdTotal = TotalScores.mean(), TotalScores.std()

# Setting plot style, and defining colors as official NBA colors, for aesthetics
sns.set_style('whitegrid')
nba_blue ='#1D428A'
nba_red = '#C8102E'

# Plotting the histogram as a density with bins amounting to 5 pts/bin
sns.histplot(TotalScores,
             bins = list(range(0, 201, 5)),
             stat = 'density',
             color = nba_blue,
             label = 'Distribution of score \nPer team, per game'
             )

# Adding a KDE plot seperately in order to manipulate it easier
sns.kdeplot(TotalScores,
             color=nba_red,
             lw=3,
             label = "KDE plot"
             )

# Creating a linspace to make a normal distribution on top. Using the mean as a basis
NormSpace = np.linspace(MuTotal - 50, MuTotal + 50, 1000)

# Plotting a parametric distribution
plt.plot(
    NormSpace,
    stats.norm.pdf(NormSpace, MuTotal, SdTotal),
    color = "goldenrod",
    linestyle="--",
    lw=2,
    label="Parametric normal fit"
)

#Plotting the mean and displaying the 2 decimal value
plt.axvline(MuTotal, 
            color = nba_red, 
            linestyle = '--',
            label = f'Mean score of teams\n{MuTotal:.2f}')

# Adding a legend, and making the graph symmetric
plt.legend(fontsize = 8)
plt.xlim(MuTotal - 50, MuTotal + 50)

plt.xlabel('Scores in bins of 5')
plt.ylabel('Density')

plt.show()