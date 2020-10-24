'''
Config stuff and some basic method
'''

ABC = '0123456789abcdefghijklmnopqrstuvwyz' #Klingon ABC doesn't have X. 
#ABC = '0123456789abcdefghijklmnopqrstuvwxyz'

def toInt(c):
    return ABC.find(c)

def fromInt(i):
    return ABC[i]

DATA_DIR = 'images/char-4-epoch-1/test'  # 302410 images. validate accuracy: 98.8%
H, W, C = 60, 160, 3
N_LABELS = len(ABC)
D = 4