import warnings
import sys
import os
import argparse

def str2bool(s):
    # This is for boolean type in the parser
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(s):
    # Has to be a list of str
    sub = s[1:len(s) - 1]
    l = []
    first = True
    tamp = 0
    for i, c in enumerate(sub):
        if c == ",":
            continue
        elif c == "'":
            if first:
                tamp = i + 1
                first = False
            else:
                l.append(sub[tamp:i])
                first = True
    return l

# Disable printing
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    warnings.filterwarnings("ignore")

# Restore printing
def enablePrint():
    sys.stdout = sys.__stdout__
    warnings.filterwarnings("default")

# ignore deprecation warnings
def ignore_depreciation():
    warnings.filterwarnings("ignore", category=DeprecationWarning)