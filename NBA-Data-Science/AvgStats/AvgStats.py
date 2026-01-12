import pandas as pd
from pathlib import Path
import sys
import numpy as np

ParentDir = Path(__file__).resolve().parent.parent
PullDir = ParentDir / 'PullData'

sys.path.insert(0, str(PullDir))

import PullDataNBA as pull # type: ignore

try:
    print('Downloading updated data from Kaggle...')
    df = pull.KaggleDownload('PlayerStatistics.csv')

except:
    print('Warning: The program was not able to pull updated data.\nThe following analysis is based on old data that is locally stored')
    df = pd.read_csv(ParentDir / 'Data' / 'PlayerStatistics.csv', low_memory = False)

def AvgPlayerStats(FirstName, LastName):
    funcDf = df[df['gameDateTimeEst'] >= '2025-10-01']
    
    funcDf = funcDf[funcDf['firstName']==FirstName]
    funcDf = funcDf[funcDf['lastName']==LastName]

    DropCols = ['firstName', 'lastName', 'personId', 'gameId', 'gameDateTimeEst', 'playerteamCity', 'playerteamName', 'opponentteamCity',
                'opponentteamName', 'gameType', 'gameLabel', 'gameSubLabel', 'seriesGameNumber', 'win',
                'home', 'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage']

    funcDf = funcDf.drop(columns = DropCols)

    for cols in funcDf:
        funcDf[cols] = np.mean(funcDf[cols])
    
    row = funcDf.iloc[0]

    print(f'{FirstName} {LastName} has the following averages for the 25/26 season:')
    for col, val in row.items():
        print(f'{col}: {round(val, 2)}')

print(AvgPlayerStats('Russell', 'Westbrook'))