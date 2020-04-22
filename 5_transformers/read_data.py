import pandas as pd


def read_data(filename):
    with open(filename) as f:
        data = {}
        headers = f.readline().strip().split("\t")
        for h in headers:
            data[h] = []
        line = f.readline()
        while line:
            pos = line.find("\t")
            data[headers[0]].append(line[:pos])
            data[headers[1]].append(line[pos+1:-1])
            line = f.readline()
    return pd.DataFrame(data)
