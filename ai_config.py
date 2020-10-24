'''
Config stuff and some basic method
'''

ABC = '01234567789abcdefghijklmnopqrstuvwxyz'

def toInt(c):
    return ABC.find(c)

def fromInt(i):
    return ABC[i]

DATA_DIR = 'images/char-4-epoch-60/train'  # 302410 images. validate accuracy: 98.8%
H, W, C = 100, 120, 3
N_LABELS = 10
D = 4