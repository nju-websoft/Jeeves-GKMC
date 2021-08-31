

import math
stopword = ["。","，","“","”","（","）","、", " ", "的", '是']
N = 5607951  # the total number of ducuments
l_avg = 49.356430182788685  # the average length of documents
k1 = 1.2
k2 = 1
b = 0.75
# def read_stopwords():
#     stopwords = []
#     fr = open('./data/stopwords.txt', 'r', encoding='utf8')
#     lines = fr.readlines()
#     for line in lines:
#         stopwords.append(line.strip())
#     return stopwords


def get_idf(word, index_map):
    if word in index_map:
        n = index_map[word]
        idf_numerator = N - n + 0.5
        idf_denominator = n + 0.5
        idf = idf_numerator / idf_denominator
        idf = math.log(1 + idf)
        return idf
    else:
        return 0.0


def get_bm25_score_word(word, question_list, content_cut, index_map):
    score = 0
    if word in stopword:
        return score
    l = len(content_cut)  # 文档长度
    # qf = question_list.count(word)
    # index_map = pindex_reverse_index_word_list(word)
    # n = len(index_map) # 包含词的文档总数
    # f = index_map[p_id] # 词频
    f = content_cut.count(word)
    # idf
    idf = get_idf(word, index_map)
    K = k1 * ((1 - b) + b * (l/l_avg))
    # r = ((f*(k1+1))/(f+K)) * ((k2+1)/(qf+k2))
    r = (f / (f + K))
    if word in content_cut:
        score = idf * r
    return score
