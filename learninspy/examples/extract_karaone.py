__author__ = 'leferrad'

from scipy.io import loadmat

path = '/media/leeandro04/Data/Downloads/ImaginedSpeechToronto/Data'
folders = ['MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15', 'MM18', 'MM20']

for fold in folders:
    file = path+'/'+fold+'/all_features_simple'
    mat = loadmat(file, squeeze_me=True, struct_as_record=False)


