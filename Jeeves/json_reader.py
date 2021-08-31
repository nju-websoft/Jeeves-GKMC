import json
from feature import InputFeature, ChoiceFeature, BM25Feature
from bm25_score import get_bm25_score_word


def clean(content):
    # stopword = [" "]
    # for s in stopword:
    #     content = content.replace(s, "")
    content = content.replace(u'\u3000', u'')
    return content


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()



def _truncate_seq_rule(tokens_a, tokens_b, tokens_c, tokens_d, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c) + len(tokens_d)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) // 2:
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_d):
            tokens_c.pop()
        else:
            tokens_d.pop()

def readSample(path):
    f = open(path, encoding='utf8')
    json_List = json.load(f)
    questionIDs = []
    scenarios = []
    questions = []
    a_list = []
    b_list = []
    c_list = []
    d_list = []
    answers = []
    A_paragraphs, B_paragraphs, C_paragraphs, D_paragraphs = [], [], [], []

    for i, sample in enumerate(json_List):

        if 'id' in sample:
            questionIDs.append(sample['id'])
        else:
            questionIDs.append(sample['q_id'])
        scenarios.append(clean(sample['scenario_cut']))
        questions.append(clean(sample['question_cut']))
        a_list.append(clean(sample['A_cut']))
        b_list.append(clean(sample['B_cut']))
        c_list.append(clean(sample['C_cut']))
        d_list.append(clean(sample['D_cut']))
        A_paragraphs.append(sample['paragraph_a'])
        B_paragraphs.append(sample['paragraph_b'])
        C_paragraphs.append(sample['paragraph_c'])
        D_paragraphs.append(sample['paragraph_d'])
        answers.append(sample['answer'])

    return questionIDs, scenarios, questions, a_list, b_list, c_list, d_list, \
           A_paragraphs, B_paragraphs, C_paragraphs, D_paragraphs, answers




def seq2SubWord(sequence, tokenizer):
    tokens = sequence.split(' ')
    token_bert_list = []
    segment_label = []
    for index, token in enumerate(tokens):
        token_bert = tokenizer.tokenize(token)
        token_bert_list += token_bert
        token_bert_len = len(token_bert)
        for i in range(token_bert_len):
            segment_label.append(index)
    return tokens, token_bert_list, segment_label


sentence_tokens = ['。', '；', '；', ',','，', '．']
# sentence_tokens = ['。', '；', '；']

def getFeature(paragraph, background, question, option, tokenizer, max_seq_length):
    if question == '':
        question = '-'
    # bert tokenizer的结果
    paragraph_token, paragraph_token_bert, paragraph_segment_label = seq2SubWord(paragraph, tokenizer)
    scenario_token, scenario_token_bert, scenario_segment_label = seq2SubWord(background, tokenizer)
    question_token, question_token_bert, question_segment_label = seq2SubWord(question, tokenizer)
    option_token, option_token_bert, option_segment_label = seq2SubWord(option, tokenizer)

    # 裁剪到最大长度范围内
    _truncate_seq_rule(scenario_token_bert, paragraph_token_bert, question_token_bert, option_token_bert,
                       max_seq_length - 5)
    _truncate_seq_rule(scenario_segment_label, paragraph_segment_label, question_segment_label, option_segment_label,
                       max_seq_length - 5)
    tokens_bert = ["[CLS]"] + paragraph_token_bert + ["[SEP]"] + scenario_token_bert + ["[SEP]"] \
                  + question_token_bert + ["[SEP]"] + option_token_bert + ["[SEP]"]

    # 转化分词信息格式
    orig_to_token_split_idx = []
    # 对应的词 和 对应词级别是否为关键词的标签
    orig_to_token_split = []  # 对应的词

    offset = 1
    beg_index = 0
    pre_seg_tag = paragraph_segment_label[0]
    for i, segment_label in enumerate(paragraph_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset + beg_index, offset + i - 1))
            word = paragraph_token[pre_seg_tag]
            orig_to_token_split.append(word)
            beg_index = i
            pre_seg_tag = segment_label

    orig_to_token_split_idx.append((offset + beg_index, offset + len(paragraph_segment_label) - 1))
    orig_to_token_split.append(paragraph_token[pre_seg_tag])

    offset = offset + len(paragraph_segment_label) + 1
    beg_index = 0
    pre_seg_tag = scenario_segment_label[0]
    for i, segment_label in enumerate(scenario_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset + beg_index, offset + i - 1))
            word = scenario_token[pre_seg_tag]
            orig_to_token_split.append(word)
            beg_index = i
            pre_seg_tag = segment_label

    orig_to_token_split_idx.append((offset + beg_index, offset + len(scenario_segment_label) - 1))
    orig_to_token_split.append(scenario_token[pre_seg_tag])

    offset = offset + len(scenario_segment_label) + 1
    beg_index = 0
    pre_seg_tag = question_segment_label[0]
    for i, segment_label in enumerate(question_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset + beg_index, offset + i - 1))
            word = question_token[pre_seg_tag]
            orig_to_token_split.append(word)
            beg_index = i
            pre_seg_tag = segment_label
    orig_to_token_split_idx.append((offset + beg_index, offset + len(question_segment_label) - 1))
    orig_to_token_split.append(question_token[pre_seg_tag])

    offset = offset + len(question_segment_label) + 1
    beg_index = 0
    pre_seg_tag = option_segment_label[0]
    for i, segment_label in enumerate(option_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset + beg_index, offset + i - 1))
            word = option_token[pre_seg_tag]
            orig_to_token_split.append(word)
            beg_index = i
            pre_seg_tag = segment_label
    orig_to_token_split_idx.append((offset + beg_index, offset + len(option_segment_label) - 1))
    orig_to_token_split.append(option_token[pre_seg_tag])

    if len(orig_to_token_split_idx) != len(orig_to_token_split):
        print('len(orig_to_token_split):', len(orig_to_token_split))
        print('no_pad_split_len:', len(orig_to_token_split_idx))
    assert len(orig_to_token_split_idx) == len(orig_to_token_split)

    # 填充
    while len(orig_to_token_split_idx) < max_seq_length:
        orig_to_token_split_idx.append((-1, -1))

    # 转化为ID，并填充
    paragraph_len = len(paragraph_token_bert)
    background_len = len(scenario_token_bert)
    q_a_len = len(question_token_bert + option_token_bert)
    segment_ids = [0] * (paragraph_len + 2) + [1] * (background_len+1 + q_a_len + 2)

    input_ids = tokenizer.convert_tokens_to_ids(tokens_bert)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    para_sub_end = len(paragraph_token_bert)
    scena_sub_beg = para_sub_end + 2
    scena_sub_end = scena_sub_beg + len(scenario_token_bert) - 1
    ques_sub_beg = scena_sub_end + 2
    ques_sub_end = ques_sub_beg + len(question_token_bert) - 1
    opt_sub_beg = ques_sub_end + 2
    opt_sub_end = opt_sub_beg + len(option_token_bert) - 1

    choice_feature = ChoiceFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        orig_to_token_split_idx=orig_to_token_split_idx,  # sub_word 上的起止位置 拼接成 word
        orig_to_token_split=orig_to_token_split,  # sub_word 拼接后对应的词
        word_set=None,  # word 集合
        word_set_idx=None,  # 每个wrod出现的位置（1到多个）
        paragraph_range_subword=(1, para_sub_end),
        scenario_range_subword=(scena_sub_beg, scena_sub_end),  # 词级别段落的起止位置（beg, end）
        question_range_subword=(ques_sub_beg, ques_sub_end),
        option_range_subword=(opt_sub_beg, opt_sub_end),
        scenario=paragraph,
        question=question,
        option=option)
    return choice_feature

def getPreFinetuneChoiceFeature(paragraph, question, option, tokenizer, max_seq_length):
    p_token = tokenizer.tokenize(paragraph)
    q_token = tokenizer.tokenize(question)
    o_token = tokenizer.tokenize(option)
    _truncate_seq_pair(p_token, q_token, o_token, max_seq_length-4)
    tokens_bert = ["[CLS]"] + p_token + ["[SEP]"] + q_token + ["[SEP]"] + o_token + ["[SEP]"]
    segment_ids = [0] * (len(p_token) + 2) + [1] * (len(q_token) + len(o_token) + 2)
    input_ids = tokenizer.convert_tokens_to_ids(tokens_bert)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding
    choice_feature = ChoiceFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        orig_to_token_split_idx=None,  # sub_word 上的起止位置 拼接成 word
        orig_to_token_split=None,  # sub_word 拼接后对应的词
        word_set=None,  # word 集合
        word_set_idx=None,  # 每个wrod出现的位置（1到多个）
        paragraph_range_subword=None,
        option_range_subword=None,  # 词级别段落的起止位置（beg, end）
        question_range_subword=None,
        paragraph=paragraph,
        scenario=None,
        question=question,
        option=option
    )
    return choice_feature

def getChoiceSampleFeature_nopara(scenario, question, option, tokenizer, max_seq_length):

    if question == "":
        question = "-"
    scenario_token, scenario_token_bert, scenario_segment_label = seq2SubWord(scenario, tokenizer)
    question_token, question_token_bert, question_segment_label = seq2SubWord(question, tokenizer)
    option_token, option_token_bert, option_segment_label = seq2SubWord(option, tokenizer)

    # 裁剪到最大长度范围内
    _truncate_seq_pair(scenario_token_bert, question_token_bert, option_token_bert, max_seq_length - 4)
    _truncate_seq_pair(scenario_segment_label, question_segment_label, option_segment_label, max_seq_length - 4)
    tokens_bert = ["[CLS]"] + scenario_token_bert + ["[SEP]"] + question_token_bert + ["[SEP]"] + option_token_bert + [
        "[SEP]"]
    # 转化分词信息格式
    orig_to_token_split_idx = []
    # 对应的词 和 对应词级别是否为关键词的标签
    orig_to_token_split = []  # 对应的词
    segment_keyword_labels = []  # 对应词级别是否为关键词的标签

    offset = 1
    beg_index = 0
    pre_seg_tag = scenario_segment_label[0]
    scenario_word = []
    for i, segment_label in enumerate(scenario_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset+beg_index, offset+i-1))
            word = scenario_token[pre_seg_tag]
            orig_to_token_split.append(word)
            scenario_word.append(word)
            beg_index = i
            pre_seg_tag = segment_label

    orig_to_token_split_idx.append((offset + beg_index, offset + len(scenario_segment_label) - 1))
    orig_to_token_split.append(scenario_token[pre_seg_tag])
    scenario_word.append(scenario_token[pre_seg_tag])

    offset = offset + len(scenario_segment_label) + 1
    beg_index = 0
    question_word = []
    pre_seg_tag = question_segment_label[0]
    for i, segment_label in enumerate(question_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset + beg_index, offset + i - 1))
            orig_to_token_split.append(question_token[pre_seg_tag])
            question_word.append(question_token[pre_seg_tag])
            beg_index = i
            pre_seg_tag = segment_label
    orig_to_token_split_idx.append((offset + beg_index, offset + len(question_segment_label) - 1))
    orig_to_token_split.append(question_token[pre_seg_tag])
    question_word.append(question_token[pre_seg_tag])

    offset = offset + len(question_segment_label) + 1
    beg_index = 0
    pre_seg_tag = option_segment_label[0]
    option_word = []
    for i, segment_label in enumerate(option_segment_label):
        if segment_label != pre_seg_tag:
            orig_to_token_split_idx.append((offset + beg_index, offset + i - 1))
            orig_to_token_split.append(option_token[pre_seg_tag])
            option_word.append(option_token[pre_seg_tag])
            beg_index = i
            pre_seg_tag = segment_label

    orig_to_token_split_idx.append((offset + beg_index, offset + len(option_segment_label) - 1))
    orig_to_token_split.append(option_token[pre_seg_tag])
    option_word.append(option_token[pre_seg_tag])

    if len(orig_to_token_split_idx) != len(orig_to_token_split):
        print('len(orig_to_token_split):',len(orig_to_token_split))
        print('no_pad_split_len:', len(orig_to_token_split_idx))
    assert len(orig_to_token_split_idx) == len(orig_to_token_split)

    word_set = set(orig_to_token_split)
    word_set = list(word_set)
    word_set.sort(key=orig_to_token_split.index)
    word_set_idx = []
    for item in word_set:
        word_set_idx_item = []
        for i, (t, t_idx) in enumerate(zip(orig_to_token_split, orig_to_token_split_idx)):
            if t == item:
                word_set_idx_item += t_idx
        word_set_idx.append(word_set_idx_item)

    # 填充
    while len(orig_to_token_split_idx) < max_seq_length:
        orig_to_token_split_idx.append((-1, -1))

    # 填充
    while len(segment_keyword_labels) < max_seq_length:
        segment_keyword_labels.append(-1)

    # 转化为ID，并填充
    background_len = len(scenario_token_bert)
    q_a_len = len(question_token_bert + option_token_bert)
    segment_ids = [0] * (background_len + 2) + [1] * (q_a_len + 2)

    input_ids = tokenizer.convert_tokens_to_ids(tokens_bert)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    scen_sub_end = len(scenario_token_bert)
    ques_sub_beg = scen_sub_end + 2
    ques_sub_end = ques_sub_beg + len(question_token_bert) - 1
    opt_sub_beg = ques_sub_end + 2
    opt_sub_end = opt_sub_beg + len(option_token_bert) - 1
    option_feature = ChoiceFeature(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        orig_to_token_split_idx=orig_to_token_split_idx,  # sub_word 上的起止位置 拼接成 word
        orig_to_token_split=orig_to_token_split,  # sub_word 拼接后对应的词
        word_set=word_set,# word 集合
        word_set_idx=word_set_idx,# 每个wrod出现的位置（1到多个）
        paragraph_range_subword=None,
        scenario_range_subword=(1, scen_sub_end),  # 词级别段落的起止位置（beg, end）
        question_range_subword=(ques_sub_beg, ques_sub_end),
        option_range_subword=(opt_sub_beg, opt_sub_end),
        scenario=scenario,
        question=question,
        option=option)
    return option_feature, scenario_word, question_word, option_word


def getBM25Feature(tokens_query, paragraphs, max_seq_length, index_map):
    features = []
    has_positive_label = 0
    for index, paragraph_item in enumerate(paragraphs):
        p_id = paragraph_item['p_id']
        paragraph = paragraph_item['paragraph_cut']
        if 'answerable' not in paragraph_item:
            p_labels = 0
        else:
            p_labels = paragraph_item['answerable']
        if p_labels > 0:
            has_positive_label = 1
        tokens_paragraph = paragraph.split(' ')
        bm25_list = []
        for q in tokens_query:
            score = get_bm25_score_word(q, tokens_query, tokens_paragraph, index_map)
            bm25_list.append(score)

        while(len(bm25_list) < max_seq_length):
            bm25_list.append(0)
        features.append(
            BM25Feature(
                bm25_score=bm25_list,
                p_labels=p_labels,
                paragraph=paragraph,
                p_id=p_id
            )
        )
    return features, has_positive_label


def getSampleFeature(tokenizer, max_seq_length, dataset_path, index_map, arg_stopwords, arg_negative_words):
    stopwords, negative_words = arg_stopwords, arg_negative_words
    questionIDs, scenarios, questions, a_list, b_list, c_list, d_list, \
    A_paragraphs, B_paragraphs, C_paragraphs, D_paragraphs, answers = readSample(dataset_path)
    features = []
    count_i = 0
    for q_id, scenario, question, a, b, c, d, A_paragraph, B_paragraph, C_paragraph, D_paragraph, answer \
            in zip(questionIDs, scenarios, questions, a_list, b_list, c_list, d_list,
                   A_paragraphs, B_paragraphs, C_paragraphs, D_paragraphs, answers):
        # if count_i > 20:
        #     break
        count_i += 1
        have_negative_word = 1
        for word in negative_words:
            if word in question:
                have_negative_word = -1
                break

        # 只有题目的格式
        feature_a_nopara, b_word_a, q_word_a, o_word_a = getChoiceSampleFeature_nopara(scenario, question, a, tokenizer,
                                                             max_seq_length)
        feature_b_nopara, b_word_b, q_word_b, o_word_b = getChoiceSampleFeature_nopara(scenario, question, b, tokenizer,
                                                             max_seq_length)
        feature_c_nopara, b_word_c, q_word_c, o_word_c = getChoiceSampleFeature_nopara(scenario, question, c, tokenizer,
                                                             max_seq_length)
        feature_d_nopara, b_word_d, q_word_d, o_word_d = getChoiceSampleFeature_nopara(scenario, question, d, tokenizer,
                                                             max_seq_length)
        tokens_a = feature_a_nopara.word_set
        feature_bm25_a, have_p_labels_a = getBM25Feature(tokens_a, A_paragraph, max_seq_length, index_map)
        tokens_b = feature_b_nopara.word_set
        feature_bm25_b, have_p_labels_b = getBM25Feature(tokens_b, B_paragraph, max_seq_length, index_map)
        tokens_c = feature_c_nopara.word_set
        feature_bm25_c, have_p_labels_c = getBM25Feature(tokens_c, C_paragraph, max_seq_length, index_map)
        tokens_d = feature_d_nopara.word_set
        feature_bm25_d, have_p_labels_d = getBM25Feature(tokens_d, D_paragraph, max_seq_length, index_map)

        bm25_feature = feature_bm25_a + feature_bm25_b + feature_bm25_c + feature_bm25_d
        choice_features = [feature_a_nopara] + [feature_b_nopara] + [feature_c_nopara] + [feature_d_nopara]
        features.append(
            InputFeature(
                q_id=q_id,
                option_features=choice_features,
                bm25_feature=bm25_feature,
                answer=answer,
                have_negative_word=have_negative_word,
                have_p_labels=[have_p_labels_a, have_p_labels_b, have_p_labels_c, have_p_labels_d]
            )
        )
    return features

def getSampleFeaturePreFinetune(tokenizer, max_seq_length, dataset_path):
    f = open(dataset_path, encoding='utf8')
    json_List = json.load(f)
    features = []
    for sample in json_List:
        questionIDs = sample['id']
        question = clean(sample['question'])
        a = clean(sample['A'])
        b = clean(sample['B'])
        c = clean(sample['C'])
        d = clean(sample['D'])
        A_paragraph = sample['paragraph_a'][0]['paragraph']
        B_paragraph = sample['paragraph_b'][0]['paragraph']
        C_paragraph = sample['paragraph_c'][0]['paragraph']
        D_paragraph = sample['paragraph_d'][0]['paragraph']
        answer = sample['answer']

        choice_feature_a = getPreFinetuneChoiceFeature(A_paragraph, question, a, tokenizer, max_seq_length)
        choice_feature_b = getPreFinetuneChoiceFeature(B_paragraph, question, b, tokenizer, max_seq_length)
        choice_feature_c = getPreFinetuneChoiceFeature(C_paragraph, question, c, tokenizer, max_seq_length)
        choice_feature_d = getPreFinetuneChoiceFeature(D_paragraph, question, d, tokenizer, max_seq_length)
        features.append(
            InputFeature(
                q_id=questionIDs,
                option_features=[choice_feature_a, choice_feature_b, choice_feature_c, choice_feature_d],
                answer=answer
            )
        )
    return features


stopwords = None
negative_words = None