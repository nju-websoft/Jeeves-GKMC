class InputFeature(object):
    def __init__(self,
                 q_id,
                 option_features, # [feature_a, feature_b, feature_c, feature_d]
                 answer,
                 bm25_feature=None,
                 have_negative_word=None,
                 have_p_labels=None):
        self.q_id = q_id
        self.option_features = option_features
        self.bm25_feature = bm25_feature
        self.answer = answer
        self.have_negative_word = have_negative_word
        self.have_p_labels = have_p_labels

class BM25Feature(object):
    def __init__(self,
                 bm25_score,
                 p_labels,
                 paragraph,
                 p_id=None):
        self.p_id = p_id
        self.bm25_score = bm25_score
        self.p_labels = p_labels
        self.paragraph = paragraph

class ChoiceFeature(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 orig_to_token_split_idx=None,  # sub_word 上的起止位置 拼接成 word
                 orig_to_token_split=None,  # sub_word 拼接后对应的词
                 word_set=None, # word 集合
                 word_set_idx=None, # 每个wrod出现的位置（1到多个）
                 paragraph_range_subword=None,
                 scenario_range_subword=None,  # 词级别段落的起止位置（beg, end）
                 question_range_subword=None,
                 option_range_subword=None,
                 paragraph=None,
                 scenario=None,
                 question=None,
                 option=None,
                 have_p_labels=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.orig_to_token_split_idx = orig_to_token_split_idx
        self.orig_to_token_split = orig_to_token_split
        self.word_set = word_set
        self.word_set_idx = word_set_idx
        self.scenario_range_subword = scenario_range_subword # 词级别段落的起止位置（beg, end）
        self.question_range_subword = question_range_subword
        self.option_range_subword = option_range_subword
        self.paragraph = paragraph
        self.scenario = scenario
        self.question = question
        self.option = option
        self.paragraph_range_subword = paragraph_range_subword
        self.have_p_labels = have_p_labels
