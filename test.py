import scandir
import os, sys
import time

import nltk

def find(name, path):
    for root, dirs, files in scandir.walk(path):
        if root.endswith(name):
            return root

def find_nltk_data():
    start = time.time()
    path_to_nltk_data = find('nltk_data', '/')
    print >> sys.stderr, 'Finding nltk_data took', time.time() - start
    print >> sys.stderr,  'nltk_data at', path_to_nltk_data
    with open('where_is_nltk_data.txt', 'w') as fout:
        fout.write(path_to_nltk_data)
    return path_to_nltk_data

def magically_find_nltk_data():
    if os.path.exists('where_is_nltk_data.txt'):
        with open('where_is_nltk_data.txt') as fin:
            path_to_nltk_data = fin.read().strip()
        if os.path.exists(path_to_nltk_data):
            nltk.data.path.append(path_to_nltk_data)
        else:
            nltk.data.path.append(find_nltk_data())
    else:
        path_to_nltk_data  = find_nltk_data()
        nltk.data.path.append(path_to_nltk_data)


magically_find_nltk_data()
print(nltk.pos_tag('this is a foo bar'.split()))
