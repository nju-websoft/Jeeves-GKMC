# import pkuseg
# seg = pkuseg.pkuseg()
# def cut(content):
#     cut_tokens = []
#     text = seg.cut(content)
#     for item in text:
#         cut_tokens.append(item)
#     return cut_tokens
import jieba
import jieba.posseg as postag

def cut(content):
    cut_tokens = []
    jiebaList = jieba.cut(content)
    for item in jiebaList:
        cut_tokens.append(item)
    return cut_tokens

def read_negative_words():
    negative_words = []
    fr = open('./data/NegativeWord.txt', 'r', encoding='utf8')
    lines = fr.readlines()
    for line in lines:
        negative_words.append(line.strip())
    return negative_words


def read_stopwords():
    stopwords = []
    fr = open('./data/stopwords.txt', 'r', encoding='utf8')
    lines = fr.readlines()
    for line in lines:
        stopwords.append(line.strip())
    return stopwords

def posseg(content):
    words = postag.cut(content)
    words_cut = []
    pos_cut = []
    for w in words:
        words_cut.append(w.word)
        pos_cut.append(w.flag)
    return words_cut, pos_cut

if __name__ == '__main__':
    words_cut, pos_cut = posseg('北京是温带季风气候')
    print(words_cut)
    print(pos_cut)
