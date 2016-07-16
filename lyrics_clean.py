from numba.decorators import jit, autojit


def find_index(string, char):
    index = -1
    for count in range(len(string)):
        if string[count] == char:
            index = count
            break
    return index
find_index = autojit(find_index)

def string_split(string, key):
    i = find_index(string, key)
    return string[:i], string[i + 1:]
string_split = autojit(string_split)

def to_time(string):
    m, s = string_split(string, ':')
    return float(m) * 60 + float(s)
to_time = autojit(to_time)

def parse_sentence(line):
    line = str(line)
    # print(line)
    end_p = find_index(line, ']')
    if end_p == -1:
        return ''
    time = line[find_index(line, '[') + 1:end_p]
    start, end = string_split(time, ' ')
    start_time = to_time(start)
    end_time = to_time(end)
    type_ = int(line[find_index(line, '>') - 1])
    sentence = line[find_index(line, '>') + 1:]
    return start_time,end_time,type_,sentence
parse_sentence = autojit(parse_sentence)

def to_dict(string):
    import ast
    return ast.literal_eval(string)

def lyrics_clean(lyrics):
    import pandas as pd
    sentences = []
    for lyric in lyrics:
        if(lyric != '\n'):
            sen = list(parse_sentence(lyric))
            if len(sen) == 4:
                sen[3]=sen[3].replace('\r','')
                sen[3]=sen[3].replace('\\','')
                sentences.append(sen)
    return pd.DataFrame(sentences,columns=['start','end','type','sentence'])
