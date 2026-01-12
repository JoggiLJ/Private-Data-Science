from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from pathlib import Path

# This function aims at downloading updated files from the Kaggle dataset used for this project
# For may types of anlyses, this isn't necessary, but it is a proof of concept, 
# and it's important for real-time analyses.

# I am very much aware that the code may be inefficient and that the data pipeline is not ideal,
# but that is not my aim for the time being.


# First we define the function, so it may be called in other files.
# The function takes the name of the csv we want to examine, as a string
def KaggleDownload(FileName):
    
    # Detine the name of the dataset we are pulling from. Thank you to eionamoore!!!
    Dataset = 'eoinamoore/historical-nba-data-and-player-box-scores'

    # Define the directories to the Data-folder within this folder in a way that works wherever this
    # 'NBA-Data-Science' folder is downloaded
    parent_dir = Path(__file__).resolve().parent.parent
    data_dir = parent_dir / 'Data'
    data_dir.mkdir(exist_ok=True)

    # Format the file name and file directory properly
    CSVName = f'{FileName}.csv'
    CSVPath = data_dir / CSVName

    # Make sure to delete old files, to avoid duplicates
    if CSVPath.exists():
        CSVPath.unlink()

    # Initiate and authenticate Kaggle
    api = KaggleApi()
    api.authenticate()

    # Download the files from the Kaggle dataset and unzipping them 
    api.dataset_download_files(
        Dataset, 
        path = data_dir,
        unzip = True
    )

    # read the requested file using pandas, turning off lowmemory to avoid dtype warning
    df = pd.read_csv(CSVPath, low_memory = False)

    # Return the requested dataframe
    return df