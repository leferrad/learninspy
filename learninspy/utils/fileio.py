__author__ = 'leferrad'

# Dependencias externas
from scipy.io import loadmat, savemat

# Librerias internas
from ..context import sc

text_extensions = ['.dat', '.txt', '.csv']


def parse_point(line):
    # TODO dar posibilidad de cambiar separador
    values = [float(x) for x in line.split(';')]
    return values[-1], values[0:-1]


# Checks
def is_text_file(path):
    return any([path.lower().endswith(ext) for ext in text_extensions])

def is_mat_file(path):
    return path.lower().endswith('.mat')


# Loader TODO mejorar manejo de .mat
def load_file(path, varname=None):
    if is_text_file(path):
        dataset = sc.textFile(path).map(parse_point)
    elif is_mat_file(path):
        dataset = loadmat(path)[varname]
    else:
        dataset = sc.binaryFiles(path).map(parse_point)
    return dataset

# Saver TODO mejorar esto que ni anda
def save_file(data, path):
    if is_text_file(path):
        data.saveAsTextFile(path+'.txt')
    elif is_mat_file(path):
        savemat(path, {'dataset': data.collect()})
    else:
        data.saveAsPickleFile(path)
    return
