from importlib.resources import files
from os import listdir


def getExampleBlobDataPath():
    """Get path to an example csv file (circle distribution).

    Returns
    -------
    csvFilePath : str
        Path to csv file.
    """
    csvFilePath = str(files('iteraa.data').joinpath('blob.csv'))
    return csvFilePath


def getExampleSquareDataPath():
    """Get path to an example csv file (uniform square distribution).

    Returns
    -------
    csvFilePath : str
        Path to csv file.
    """
    csvFilePath = str(files('iteraa.data').joinpath('uniformSquare.csv'))
    return csvFilePath


def getCaseStudyDataPaths():
    """Get paths to the csv files used for case study (mono-, bi-, and trimetallic nanoparticles).

    Returns
    -------
    csvFilePaths : list of str
        Paths to csv files.
    """
    dataDir = str(files('iteraa.data'))
    csvFilePaths = []
    for fileName in listdir(dataDir):
        if 'caseStudy' in fileName:
            csvFilePaths.append(f"{dataDir}/{fileName}")
    return csvFilePaths

