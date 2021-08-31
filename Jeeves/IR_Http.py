import os.path
import jieba
import pymysql
from gaokao_util import cut
import requests
import json
import heapq
db = pymysql.connect("localhost", "root", "12345", "gaokao")

cursor = db.cursor()
ip = 'http://localhost:8080/'

def bm25_search(query, max_result_num):

    query_cut = cut(query)
    postdatas = {'query': query_cut, 'MaxResults':max_result_num}
    response = requests.post(ip+'corpusRetrievalBM25SegmentAPI', data=postdatas)
    contents = []
    p_ids = []
    scores = []
    results = response.content.decode('utf8')
    results = json.loads(results)
    for result in results:
        p_ids.append(int(result['id']))
        scores.append(float(result['score']))
        contents.append(result['content'])
    return p_ids, contents, scores

# 带有词权重的BM25检索
def bm25_search_weight(query_words, weight_words, max_result_num=None):

    postdatas = {'query': query_words, 'weight': weight_words, 'MaxResults': max_result_num}
    response = requests.post(ip + 'corpusRetrievalBM25WeightSegmentAPI', data=postdatas)
    contents = []
    p_ids = []
    scores = []
    results = response.content.decode('utf8')
    # print(results)
    results = json.loads(results)
    for result in results:
        p_ids.append(int(result['id']))
        scores.append(float(result['score']))
        contents.append(result['content'])
    return p_ids, contents, scores

# 带有词权重的BM25检索
def bm25_search_weight_list(query_words, weight_words, max_result_num=None):

    postdatas = {'query': query_words, 'weight': weight_words, 'MaxResults': max_result_num}
    response = requests.post(ip + 'corpusRetrievalBM25WeightSegmentListAPI', data=postdatas)
    contents = []
    p_ids = []
    scores = []
    results = response.content.decode('utf8')
    # print(results)
    results = json.loads(results)
    for result in results:
        p_ids.append(int(result['id']))
        scores.append(float(result['score']))
        contents.append(result['content'])
    return p_ids, contents, scores

def getMultiChoice(q_id):

    sql = "SELECT problem_background.background, problem.question, problem.choice_A, problem.choice_B, problem.choice_C, problem.choice_D, answer " \
          "FROM problem, problem_background WHERE problem.background_id=problem_background.id AND problem.id="+str(q_id)
    background = ""
    question = ""
    a = ""
    b = ""
    c = ""
    d = ""
    answer = ""
    try:
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        for row in results:
            background = row[0]
            question = row[1]
            a = row[2]
            b = row[3]
            c = row[4]
            d = row[5]
            answer = row[6]

    except Exception as e:
        print(e)
    return background, question, a, b, c, d, answer
