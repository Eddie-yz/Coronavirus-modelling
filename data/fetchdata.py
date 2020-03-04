import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt


def getProvinceData(province_name, csv_file="data/Wuhan-2019-nCoV.csv"):
    df = pd.read_csv(csv_file)
    df["date"] = df["date"].map(lambda x: "-".join([i.zfill(2) for i in re.split("\\D+", x)]))
    
    provinceData = df[(df["province"] == province_name) & (df['city'] != df['city'])]
    date = np.array(provinceData["date"])
    confirmed = np.array(provinceData["confirmed"])
    cured = np.array(provinceData["cured"])
    
    return date, confirmed, cured


def getCountryData(country_code, csv_file="data/Wuhan-2019-nCoV.csv"):
    df = pd.read_csv(csv_file)
    df["date"] = df["date"].map(lambda x: "-".join([i.zfill(2) for i in re.split("\\D+", x)]))
    
    countryData = df[df["countryCode"] == country_code]
    date = np.array(countryData["date"])
    confirmed = np.array(countryData["confirmed"])
    cured = np.array(countryData["cured"])

    return date, confirmed, cured



