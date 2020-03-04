import os
import requests
import csv


CSV_URL = 'https://raw.githubusercontent.com/canghailan/Wuhan-2019-nCoV/master/Wuhan-2019-nCoV.csv'

with open(os.path.split(CSV_URL)[1], 'w', newline='',encoding='utf-8') as f, \
        requests.get(CSV_URL, stream=True) as r:
    writer = csv.writer(f, delimiter=',')
    for line in r.iter_lines():
        line = line.decode('utf8')
        line = line.strip().split(',')   
        writer.writerow(line)
