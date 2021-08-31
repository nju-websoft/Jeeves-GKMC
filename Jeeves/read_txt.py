# -*- coding: utf-8 -*-
import json
import sys

def read_stopwords():
    stopwords = []
    fr = open('./data/stopwords.txt', 'r', encoding='utf8')
    lines = fr.readlines()
    for line in lines:
        stopwords.append(line.strip())
    return stopwords

def read_negative_words():
    negative_words = []
    fr = open('./data/NegativeWord.txt', 'r', encoding='utf8')
    lines = fr.readlines()
    for line in lines:
        negative_words.append(line.strip())
    return negative_words

def pindex_reverse_index_word_list():
    path = './data/idf.txt'
    f = open(path, 'r', encoding='utf8')
    lines = f.readlines()
    index_map = {}
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        if len(line) < 2:
            continue
        token = line[0]
        row = line[1]
        index_map[token] = int(row)
    f.close()
    return index_map


def read_ground_truth():
    path = './data/GeoSQAc_dataset.json'
    f = open(path, encoding='utf8')
    json_List = json.load(f)
    ground_truth_map = {}

    for question_map in json_List:
        paragraph_a = question_map['paragraph_a']
        paragraph_b = question_map['paragraph_b']
        paragraph_c = question_map['paragraph_c']
        paragraph_d = question_map['paragraph_d']
        q_id = question_map['id']
        p_ids = []
        for paragraph in paragraph_a:
            p_ids.append(paragraph['p_id'])
        if len(p_ids) > 0:
            key = str(q_id)+'-'+str(0)
            ground_truth_map[key] = p_ids

        p_ids = []
        for paragraph in paragraph_b:
            p_ids.append(paragraph['p_id'])
        if len(p_ids) > 0:
            key = str(q_id) + '-' + str(1)
            ground_truth_map[key] = p_ids

        p_ids = []
        for paragraph in paragraph_c:
            p_ids.append(paragraph['p_id'])
        if len(p_ids) > 0:
            key = str(q_id) + '-' + str(2)
            ground_truth_map[key] = p_ids

        p_ids = []
        for paragraph in paragraph_d:
            p_ids.append(paragraph['p_id'])
        if len(p_ids) > 0:
            key = str(q_id) + '-' + str(3)
            ground_truth_map[key] = p_ids
        # print(ground_truth_map)
    return ground_truth_map

if __name__ == '__main__':
    ground_truth_map = read_ground_truth()
    print(len(ground_truth_map.keys()))