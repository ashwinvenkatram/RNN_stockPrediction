import os
import sys
import pandas as pd
import shutil

# data = pd.read_csv('output_list.txt', sep=" ", header=None)
# data.columns = ["a", "b", "c", "etc."]

def getFiles():
    fileArr = []
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            if '.txt' in name:
                fileArr.append(os.path.join(root, name))
    return fileArr

def readFile(filedir):
    # with open(filedir) as infile:
    #     read_data = infile.read()
    # print(filedir)
    read_data = pd.read_csv(filedir, sep=",", header=None)
    read_data.columns = ["ticker", "date", "open", "high","low","close",'']
    read_data = read_data.drop(columns=[''])

    return read_data

def updateTickerFile(fileData):
    for index, row in fileData.iterrows():
        tickerSymbol = row.ticker
        tickerData = [row.date, row.open, row.high, row.low, row.close]

        with open("Arranged_Dataset/"+tickerSymbol+".csv","a") as outfile:
            strTickData = ','.join(map(str, tickerData))
            strTickData = strTickData + "\n"
            outfile.write(strTickData)
        outfile.close()

def updateTickerDict(fileData,NYSEDataset):
    for index, row in fileData.iterrows():
        tickerSymbol = row.ticker
        tickerData = [row.date, row.open, row.high, row.low, row.close]

        strTickData = ','.join(map(str, tickerData))
        strTickData = strTickData + "\n"

        NYSEDataset.setdefault(tickerSymbol,[]).append(strTickData)

def dumpToFiles(NYSEDataset, cleanDir):
    for key in NYSEDataset.keys():
        with open(cleanDir + "/" + key + ".csv","w") as outfile:
            for row in NYSEDataset[key]:
                outfile.write(row)
        outfile.close()

if __name__=="__main__":

    NYSEDataset = {}

    cleanDir = 'Arranged_Dataset'
    if not os.path.exists(cleanDir):
        os.makedirs(cleanDir)
        print("created Arranged Dataset directory")
    else:
        shutil.rmtree(cleanDir)
        os.makedirs(cleanDir)
        print("Snapped fingers and recreated directory")

    fileArr = getFiles()

    # For 1 file:
    # Load data per day
    # for each ticker, append to a datafile with its name
    counter = 0
    lenFiles = len(fileArr)
    print("Working...")
    for filePath in fileArr:
        fileData = readFile(filePath)
        updateTickerDict(fileData,NYSEDataset)
        print(counter,"/",lenFiles,"; updating: ",filePath)
        counter = counter + 1
    print("Dumping to files")

    dumpToFiles(NYSEDataset, cleanDir)

    print("Done")

