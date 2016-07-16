# NOTE: In this code, it will generate an edglist containing song and
# lyrics relationship.

# Pre-requirement files :
# 1. 'data/Western_songs_info.tsv' : contain the meta data of each song and their corresponding song id
# 2. 'data/Western_songs.csv' : contain the title of each song and their corresponding song id
# 3. 'data/Western_songs_lyrics.tsv' : contain the lyrics of each song. More , the lyrics are with alignment of the audio using time indexes.
# OUTPUT :
# 1. cleaned.adjlist : In the edgelist (adjlist), every index in the edge list correspond to a song node (song ID) or a vocabrary node (smaller voc index).
# If a song node is connected to a voc node , it means that the song contain the voc in its lyrics.
# If a voc node is connected to a song node, it means that the voc is contained in that song's lyrics.
# 2. voc.dict : the file containing dictionary of the whole vocabrary, the
# first maps term to index, the second maps index to term.

# XXX MAIN CODE START FROM HERE !
from __future__ import print_function
# XXX read data from file ################################################
from detecting_language import *
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing.pool import ApplyResult

song_info = [element.split('\t') for element in [str(line, 'utf8').rstrip(
    '\n') for line in open('data/Western_songs_info.tsv', 'rb')]]

song_ids = [element for element in [line.split(
    ',') for line in open('data/Western_songs.csv')][0]]

song_lyrics_tmp = [str(element, 'utf-8').split('\t') for element in open(
    'data/Western_songs_lyrics.tsv', 'rb')]

song_lyrics_data = [[element[0], element[1].split(
    '\\n')] for element in song_lyrics_tmp]

print('number of songs in Western_songs.csv', len(song_ids))
print('number of songs in Western_songs_info.csv', len(song_info))
print('number of songs in Western_songs_lyrics.tsv', len(song_lyrics_tmp))
# TODO:remove none english songs :
# detecting language
#%timeit language_tag = [get_language(''.join([sen for sen in element[1]])) for element in song_lyrics_data]
# REVIEW original: 43.7s parallelize: 30.9 s


def detect_language_tag(element):
    return get_language(''.join([sen for sen in element[1]]))

language_tag = Parallel(n_jobs=multiprocessing.cpu_count())(
    delayed(detect_language_tag)(element) for element in song_lyrics_data)


new_song_lyrics_data = [song_lyrics_data[i] for i in range(
    len(language_tag)) if language_tag[i] == 'english']
print('song number after non english songs are removed:', len(new_song_lyrics_data))

##########################################################################
import pandas as pd

song_info_table = pd.DataFrame(song_info, columns=['id', 'title', 'album', 'artist'])
song_info_table.set_index('id')
song_ids_table = pd.DataFrame(song_ids, columns=['id'])
song_ids_table

lyrics_table = pd.DataFrame(new_song_lyrics_data, columns=['id', 'lyrics'])

lyrics_table = lyrics_table.set_index('id')
lyrics_table[['lyrics']]
from lyrics_clean import *

lyrics_table=lyrics_table[['lyrics']].applymap(lyrics_clean)

for i in lyrics_table.index:
    lyrics_table.loc[i]['lyrics'].index=[i]*len(lyrics_table.loc[i]['lyrics'])

lyrics_table.iloc[1]['lyrics'][['sentence']].values.transpose().tolist()[0]

all_lyrics_table = pd.concat(lyrics_table['lyrics'].values)
# Calculate sentence similarity
from analysis_package import *
import skipthoughts
model=skipthoughts.load_model()

all_lyrics = all_lyrics_table[['sentence']].values.transpose().tolist()[0]
len(all_lyrics)
vectors = skipthoughts.encode(model,all_lyrics[:10000],use_eos=True)

lyrics_embedding_table = pd.DataFrame(vectors,index = all_lyrics[:10000])

neighbor_table = k_nearest_neighbor(lyrics_embedding_table,5)

def range_filter(df, keys, lower, upper):
    key = keys[0]
    true_false_table = np.array(df[[key]] < upper) & np.array(df[[key]] > lower)
    for key in keys:
        true_false_table = true_false_table & (np.array(df[[key]] < upper) & np.array(df[[key]] > lower))
    return df[true_false_table]
range_filter(neighbor_table,[3,5,7,9],0.,0.5)
range_filter(neighbor_table,[3,5,7],0.,0.5)
range_filter(neighbor_table,[3,5],0.,0.5)

# TODO : merge sentence with similar meaning in one lyrics
# TODO : merge sentence with similar meaning across lyrics
