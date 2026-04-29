import numpy as np

def clbtfile(clbt, filename=None):
    res = "Calibration Result:\n"
    for k, v in clbt.items():
        res += f"{k}:\n{v}\n"
    if filename:
        with open(filename, 'w') as f:
            f.write(res)
    else:
        print(res)
