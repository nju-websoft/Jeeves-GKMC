"""BERT finetuning runner."""
import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import math
import numpy as np
import json
import torch
import ast
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from model.tokenization import BertTokenizer
from model.modeling import Jeeves_Model
from model.optimization import BertAdam
from json_reader import getSampleFeature, getSampleFeaturePreFinetune
from read_txt import pindex_reverse_index_word_list, read_ground_truth, read_negative_words, read_stopwords
import heapq

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# PYTHONIOENCODING=utf-8 python run_jeeves.py

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
DEBUG = False

def grid_search(ans_a, ans_b, ans_c, answers):
    step = 0.01
    best_acc = 0
    alpha, beta, gama = 1, 0, 0
    for a in range(0, int(1/step)):
        a = a * step
        b_max = 1 - a
        for b in range(0, int(b_max/step)):
            b = b * step
            c = b_max - b
            ans = a * ans_a + b * ans_b + c * ans_c
            answer_right, answer_count = 0, 0
            ans_predict = ans.cpu().detach().numpy().tolist()
            for p, answer in zip(ans_predict, answers):
                p_ans = p.index(max(p))
                if p_ans == answer:
                    answer_right += 1
                answer_count += 1
            acc = answer_right / answer_count

            if acc > best_acc:
                alpha, beta, gama = a, b, c
                best_acc = acc
    return best_acc, alpha, beta, gama

def batch_prefinetune_feature(batch):
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    input_ids = []
    q_ids = []
    input_mask = []
    segment_ids = []
    answers = []

    for batch_example in batch:
        q_ids.append(batch_example.q_id)
        answers.append(answer_map[batch_example.answer])
        choice_features = batch_example.choice_features
        for choice_feature in choice_features:
            input_ids.append(choice_feature.input_ids)
            input_mask.append(choice_feature.input_mask)
            segment_ids.append(choice_feature.segment_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    answers = torch.tensor(answers, dtype=torch.long)

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    answers = answers.to(device)

    return input_ids, input_mask, segment_ids, answers, q_ids


def get_batch_feature(batch):
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    input_ids, input_mask, segment_ids = [], [], []
    answer, have_negative_word, q_ids, p_ids = [], [], [], []
    orig_to_token_split_idx, orig_to_token_split = [], []
    word_set, word_set_idx = [], []
    scenario_range_subword, question_range_subword, option_range_subword = [], [], []
    paragraphs, scenarios, questions, options = [], [], [], []
    bm25_feature, p_labels, have_p_labels = [], [], []

    for batch_example in batch:
        q_ids.append(batch_example.q_id)
        answer.append(answer_map[batch_example.answer])
        have_negative_word.append(batch_example.have_negative_word)
        option_features = batch_example.option_features
        for option_feature in option_features:
            input_ids.append(option_feature.input_ids)
            input_mask.append(option_feature.input_mask)
            segment_ids.append(option_feature.segment_ids)

            orig_to_token_split_idx.append(option_feature.orig_to_token_split_idx)
            orig_to_token_split.append(option_feature.orig_to_token_split)
            word_set.append(option_feature.word_set)
            word_set_idx.append(option_feature.word_set_idx)

            scenario_range_subword.append(option_feature.scenario_range_subword)
            question_range_subword.append(option_feature.question_range_subword)
            option_range_subword.append(option_feature.option_range_subword)

            scenarios.append(option_feature.scenario)
            questions.append(option_feature.question)
            options.append(option_feature.option)

        bm25_features = batch_example.bm25_feature
        have_p_labels.append(batch_example.have_p_labels)

        for feature in bm25_features:
            p_ids.append(feature.p_id)
            bm25_feature.append(feature.bm25_score)
            p_labels.append(feature.p_labels)
            paragraphs.append(feature.paragraph)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)
    answer = torch.tensor(answer, dtype=torch.long)
    have_negative_word = torch.tensor(have_negative_word, dtype=torch.long)
    bm25_feature = torch.tensor(bm25_feature, dtype=torch.float)
    p_labels = torch.tensor(p_labels, dtype=torch.long)
    have_p_labels = torch.tensor(have_p_labels, dtype=torch.long)
    p_ids = torch.tensor(p_ids, dtype=torch.long)

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    answer = answer.to(device)
    have_negative_word = have_negative_word.to(device)
    bm25_feature = bm25_feature.to(device)
    p_labels = p_labels.to(device)
    have_p_labels = have_p_labels.to(device)
    p_ids = p_ids.to(device)

    return q_ids, answer, have_negative_word, input_ids, input_mask, segment_ids, \
           orig_to_token_split_idx, orig_to_token_split, \
           scenario_range_subword, question_range_subword, option_range_subword, \
           paragraphs, scenarios, questions, options, bm25_feature, p_labels, have_p_labels, word_set, word_set_idx, p_ids

def eval(model, test_examples, alpha, beta, gama, name, save_file_name=None):
    if name == 'test':
        do_eval = True
    else:
        do_eval = False
    model.eval()
    step_count = len(test_examples) // args.eval_batch_size
    if step_count * args.eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    hit, map, ndcg, ir_count = 0, 0, 0, 0
    count = 0
    answer_right, answer_count = 0, 0
    result = []
    score_retriever_all, score_reader1_all, score_reader2_all, answer_all = None, None, None, None
    for step in step_trange:
        beg_index = step * args.eval_batch_size
        end_index = min((step + 1) * args.eval_batch_size, len(test_examples))
        batch = [example for example in test_examples[beg_index:end_index]]

        q_ids, answers, have_negative_word, input_ids, input_mask, segment_ids, \
        orig_to_token_split_idx, orig_to_token_split, \
        scenario_range_subword, question_range_subword, option_range_subword, \
        paragraphs, scenarios, questions, options, bm25_feature, p_labels, \
        have_p_labels, word_set, word_set_idx, p_ids = get_batch_feature(batch)

        with torch.no_grad():
            score_retriever_soft, score_reader1_soft, score_reader2_soft, p_labels, p_ids, z_l, paragraphs = model(input_ids, input_mask,
                                                                                                segment_ids,
                                                                                                q_ids=q_ids,
                                                                                                answers=answers,
                                                                                                have_negative_word=have_negative_word,
                                                                                                orig_to_token_split_idx=orig_to_token_split_idx,
                                                                                                orig_to_token_split=orig_to_token_split,
                                                                                                word_set_idx=word_set_idx,
                                                                                                word_set=word_set,
                                                                                                paragraphs=paragraphs,
                                                                                                scenarios=scenarios,
                                                                                                questions=questions,
                                                                                                options=options,
                                                                                                scenario_range_subword=scenario_range_subword,
                                                                                                question_range_subword=question_range_subword,
                                                                                                option_range_subword=option_range_subword,
                                                                                                bm25_feature=bm25_feature,
                                                                                                p_labels=p_labels,
                                                                                                have_p_labels=have_p_labels,
                                                                                                p_ids=p_ids,
                                                                                                do_train=False,
                                                                                                do_eval=do_eval,
                                                                                                do_pre_finetune=False)

            if score_reader1_all is None:
                score_reader1_all = score_reader1_soft
            else:
                score_reader1_all = torch.cat([score_reader1_all, score_reader1_soft], dim=0)
            if score_reader2_all is None:
                score_reader2_all = score_reader2_soft
            else:
                score_reader2_all = torch.cat([score_reader2_all, score_reader2_soft], dim=0)
            if score_retriever_all is None:
                score_retriever_all = score_retriever_soft
            else:
                score_retriever_all = torch.cat([score_retriever_all, score_retriever_soft], dim=0)

            if answer_all is None:
                answer_all = answers
            else:
                answer_all = torch.cat([answer_all, answers], dim=0)

            para_num = args.tau_num
            z_l = z_l.cpu().detach().numpy().tolist()
            p_ids = p_ids.view(-1, para_num)
            p_ids = p_ids.cpu().detach().numpy().tolist()

            score_retriever_soft = score_retriever_soft.cpu().detach().numpy().tolist()
            score_reader1_soft = score_reader1_soft.cpu().detach().numpy().tolist()
            score_reader2_soft = score_reader2_soft.cpu().detach().numpy().tolist()

            paragraphs_fold = []
            p_step = len(paragraphs) // para_num
            for i in range(p_step):
                p_item = []
                for j in range(para_num):
                    p_item.append(paragraphs[i * para_num + j])
                paragraphs_fold.append(p_item)

            result_list = save_results(q_ids, paragraphs_fold,
                                       scenarios, questions,
                                       options, z_l, 10, answers, score_retriever_soft, score_reader1_soft, score_reader2_soft, p_ids)
            result += result_list

            for i, q in enumerate(q_ids):
                for opt_index in range(4):
                    score = z_l[i * 4 + opt_index]
                    p_ids_list = p_ids[i * 4 + opt_index]
                    count += 1
                    key = str(q) + '-' + str(opt_index)
                    if key in ground_truth:
                        pid_ground_truth = ground_truth[key]
                    else:
                        pid_ground_truth = []
                    if len(pid_ground_truth) > 0:
                        data = heapq.nlargest(args.ir_metric_at_n, enumerate(score), key=lambda x: x[1])
                        max_index, vals = zip(*data)
                        max_num_index_list = list(max_index)
                        for idx_i, idx in enumerate(max_num_index_list):
                            p_id = p_ids_list[idx]
                            if p_id in pid_ground_truth:
                                hit += 1
                                break
                        ap_right_num = 0
                        ap = 0
                        for idx_i, idx in enumerate(max_num_index_list):
                            p_id = p_ids_list[idx]
                            if p_id in pid_ground_truth:
                                ap_right_num += 1
                                ap += ap_right_num / (idx_i + 1)
                        if ap_right_num > 0:
                            ap = ap / ap_right_num
                        map += ap
                        dcg = 0
                        idcg = 0

                        for idx_i, idx in enumerate(max_num_index_list):
                            p_id = p_ids_list[idx]
                            if p_id in pid_ground_truth:
                                dcg += 1 / math.log2(idx_i + 2)
                            if i < len(pid_ground_truth):
                                idcg += 1 / math.log2(i + 2)

                        ndcg += dcg/idcg
                        ir_count += 1

    if alpha is None:
        acc, alpha, beta, gama = grid_search(score_retriever_all, score_reader1_all, score_reader2_all, answer_all)

    ans_predict = alpha * score_retriever_all + beta * score_reader1_all + gama * score_reader2_all
    ans_predict = ans_predict.cpu().detach().numpy().tolist()
    for p, a in zip(ans_predict, answer_all):
        p_ans = p.index(max(p))
        if p_ans == a:
            answer_right += 1
        answer_count += 1
    acc = answer_right / answer_count

    if ir_count > 0:
        print('hit' + str(args.ir_metric_at_n) + ":", round(hit / ir_count, 4),
              'map' + str(args.ir_metric_at_n) + ":", round(map / ir_count, 4),
              'ndcg' + str(args.ir_metric_at_n) + ":", round(ndcg / ir_count, 4))

    print('acc:', round(acc, 4), 'alpha, beta, gama:', alpha, beta, gama)

    if save_file_name is not None:
        with open(save_file_name, 'w') as f:
            json.dump(result, f, ensure_ascii=False,  indent=2)
    return hit / ir_count, map/ir_count, ndcg/ir_count, acc, alpha, beta, gama


def save_results(q_ids, paragraphs, scenarios, questions, options, rank_scores, top_k, answers,
                 score_retriever_soft, score_reader1_soft, score_reader2_soft, p_ids):
    answers = answers.cpu().detach().tolist()
    option_map = {0:'A', 1:'B', 2:'C', 3:'D'}
    result_list = []
    for i, q in enumerate(q_ids):
        result = {}
        background = scenarios[i * 4]
        question = questions[i * 4]
        answer = answers[i]
        score_retriever = score_retriever_soft[i]
        score_reader1 = score_reader1_soft[i]
        score_reader2 = score_reader2_soft[i]

        result['q_id'] = q
        result['scenario'] = background.replace(' ','')
        result['question'] = question.replace(' ','')
        result['answer'] = option_map[answer]
        result['score_retriever'] = score_retriever
        result['score_reader1'] = score_reader1
        result['score_reader2'] = score_reader2


        for opt_index in range(4):
            scores = rank_scores[i * 4 + opt_index]

            p_id_list = p_ids[i * 4 + opt_index]
            paragraph_list = paragraphs[i * 4 + opt_index]
            option = options[i * 4 + opt_index].replace(' ','')
            option_str = option_map[opt_index]
            result['choice_'+option_str] = option
            data = heapq.nlargest(top_k, enumerate(scores), key=lambda x: x[1])
            max_index, vals = zip(*data)
            max_num_index_list = list(max_index)
            paragraph_infos = []
            for idx in max_num_index_list:

                p_id = p_id_list[idx]
                paragraph = paragraph_list[idx].replace(' ','')
                paragraph_score = scores[idx]
                paragraph_info = {}
                paragraph_info['p_id'] = p_id
                paragraph_info['paragraph'] = paragraph
                paragraph_info['paragraph_score'] = paragraph_score
                paragraph_infos.append(paragraph_info)
            result['paragraph_'+option_str] = paragraph_infos
        result_list.append(result)
    return result_list


def eval_pre_finetune(model, test_examples):
    model.eval()
    step_count = len(test_examples) // args.eval_batch_size
    if step_count * args.eval_batch_size < len(test_examples):
        step_count += 1
    step_trange = trange(step_count)
    right_count = 0
    count = 0
    for step in step_trange:
        beg_index = step * args.eval_batch_size
        end_index = min((step + 1) * args.eval_batch_size, len(test_examples))
        batch = [example for example in test_examples[beg_index:end_index]]
        input_ids, input_mask, segment_ids, answers, q_ids = batch_prefinetune_feature(batch)
        with torch.no_grad():
            choice_logits = model(input_ids, input_mask, segment_ids,
                                  do_train=False, do_pre_finetune=True)
        choice_logits = choice_logits.cpu().detach().numpy()
        answers = answers.cpu().detach().numpy()
        p_answer = np.argmax(choice_logits, axis=1)
        right_answer = sum(p_answer == answers)
        right_count += right_answer
        count += answers.size
    print(" accuary：" + str(round(right_count / count, 4)))
    return right_count / count


def train_pre_finetune(model, train_examples, test_examples):
    logger.info("  Num train examples = %d", len(train_examples))
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    logger.info("  Num train steps = %d", num_train_steps)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynaimc_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    order = list(range(len(train_examples)))
    # random.seed(args.seed)
    random.shuffle(order)
    model.train()
    tr_loss_choice = 0
    nb_tr_steps_choice = 0
    step_count = len(train_examples) // args.train_batch_size
    if step_count * args.train_batch_size < len(train_examples):
        step_count += 1

    right_count, count = 0, 0
    answerable_right_count, answerable_count = 0, 0
    best_accu = 0.25
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        step_trange = trange(step_count)
        for step in step_trange:
            beg_index = step * args.train_batch_size
            end_index = min((step + 1) * args.train_batch_size, len(train_examples))
            order_index = order[beg_index:end_index]
            batch = [train_examples[index] for index in order_index]

            input_ids, input_mask, segment_ids, answers, q_ids = batch_prefinetune_feature(batch)
            choice_logits, loss_choice = model(input_ids, input_mask, segment_ids,
                                               do_train=True, do_pre_finetune=True)
            if loss_choice is not None and loss_choice.item() > 0:
                tr_loss_choice += loss_choice.item()
                nb_tr_steps_choice += 1

            loss = loss_choice

            if n_gpu > 1:
                loss = loss.mean()
            if args.fp16 and args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            choice_logits = choice_logits.cpu().detach().numpy()
            answers = answers.cpu().detach().numpy()
            p_answer = np.argmax(choice_logits, axis=1)

            right_answer = sum(p_answer == answers)
            right_count += right_answer
            count += answers.size

            if count != 0 and answerable_count != 0:
                loss_show = ' Epoch:' + str(epoch) + " acc：" + str(round(right_count / count, 2))
                step_trange.set_postfix_str(loss_show)

        if args.do_pre_finetune_test:
            accu = eval_pre_finetune(model, test_examples)
            loss_show = ' Epoch:' + str(epoch) + " acc：" + str(round(right_count / count, 2))
            logger.info(loss_show)

            if accu > best_accu:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                output_model_file = '/'.join(
                    [args.output_dir, "ernie-c3-m-128-" + str(round(accu, 4)) + ".bin"])
                torch.save(model_to_save.state_dict(), output_model_file)


                print('saving current best model')
                best_accu = accu

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_dir",
                        default='./data/bert_wwm_pytorch_ch/',
                        type=str,
                        help="The BERT dreader2ctory")
    parser.add_argument("--save_name",
                        default='wwm-1',
                        type=str)
    parser.add_argument("--ON_AZURE",
                        default=False,
                        help="if training the mode on azure")
    parser.add_argument("--ON_PHILLY",
                        default=False,
                        type=bool,
                        help="if training the mode on philly")
    parser.add_argument("--results_save_path",
                        default='./results/',
                        type=str)
    parser.add_argument("--output_dir",
                        default='./outputs/',
                        type=str,
                        help="The output dreader2ctory whretriever the model checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--rev",
                        default=False,
                        type=bool,
                        help="Reverse the order of LM or not, default is False")
    parser.add_argument("--init_checkpoint",
                        default='./outputs/c3-wwm-ext-md-256-best-0.621.bin',
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model)")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        type=ast.literal_eval,
                        help="Whether to run training.")
    parser.add_argument("--do_pre_finetune",
                        default=False,
                        type=bool)
    parser.add_argument("--do_pre_finetune_test",
                        default=False,
                        type=bool)
    parser.add_argument("--do_eval_choice",
                        default=True,
                        type=ast.literal_eval)
    parser.add_argument("--do_predict",
                        default=False,
                        type=ast.literal_eval,
                        )
    parser.add_argument("--do_lucene_predict",
                        default=False,
                        type=ast.literal_eval,
                        )
    parser.add_argument("--train_dataset_path",
                        default='./data/GKMC/train_0.json',
                        type=str)
    parser.add_argument("--dev_dataset_path",
                        default='./data/GKMC/dev_0.json',
                        type=str)
    parser.add_argument("--test_dataset_path",
                        default='./data/GKMC/test_0.json',
                        type=str)
    parser.add_argument("--train_dataset_path2",
                        default=None,
                        type=str)
    parser.add_argument("--dev_dataset_path2",
                        default=None,
                        type=str)
    parser.add_argument("--test_dataset_path2",
                        default=None,
                        type=str)
    parser.add_argument("--retriever",
                        default=True,
                        type=ast.literal_eval,
                        )
    parser.add_argument("--reader1",
                        default=True,
                        type=ast.literal_eval,
                        )
    parser.add_argument("--reader2",
                        default=True,
                        type=ast.literal_eval,
                        )
    parser.add_argument("--p",
                        default=False,
                        type=ast.literal_eval,
                        )
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--choice_num",
                        default=4,
                        type=int)
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=6.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--tau_num',
                        type=int,
                        default=200)
    parser.add_argument('--k_num_train',
                        type=int,
                        default=2)
    parser.add_argument('--k_num_test',
                        type=int,
                        default=10)
    parser.add_argument('--k2_num',
                        type=int,
                        default=2)
    parser.add_argument('--ir_metric_at_n',
                        type=int,
                        default=10)
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=1,#42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        type=bool,
                        default=False,
                        # action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.ON_PHILLY:
        logger.info('Training on PHILLY')
        # BLANK because this is Microsoft Only
    elif args.ON_AZURE:
        logger.info('Training on AZURE')
        # BLANK because this is Microsoft Only
    else:
        logger.info('Training on LOCAL')


    ##########################################################################
    #  Get Machine Configuration and check input arguments
    ##########################################################################
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    # logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    ##########################################################################
    # Prepare for Model
    ##########################################################################
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=args.do_lower_case)
    logger.info('loading pretrained bert model from %s' % args.init_checkpoint)
    ground_truth = read_ground_truth()
    index_map = pindex_reverse_index_word_list()
    stopwords = read_stopwords()
    negative_words = read_negative_words()
    model = Jeeves_Model.from_pretrained(args.bert_dir, num_choices=args.choice_num, max_seq_length=args.max_seq_length,
                                         tau_num=args.tau_num, k_num_train=args.k_num_train,
                                         k_num_test=args.k_num_test, k2_num=args.k2_num, index_map=index_map,
                                         tokenizer=tokenizer)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    if args.init_checkpoint is not None:
        logger.info('loading finetuned model from %s' % args.init_checkpoint)
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata


        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')


        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        logger.info("missing keys:{}".format(missing_keys))
        logger.info('unexpected keys:{}'.format(unexpected_keys))
        logger.info('error msgs:{}'.format(error_msgs))


    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    dev_ans_best_accu = 0
    test_ir_best_hit, test_ir_best_map, test_ir_best_ndcg, test_ans_best_accu = 0, 0, 0, 0
    alpha, beta, gama = 0.3, 0.15, 0.55

    test_examples = getSampleFeature(tokenizer, args.max_seq_length, args.test_dataset_path, index_map, stopwords, negative_words)
    if args.do_predict:

        # eval(model, dev_examples, alpha, beta, gama, 'dev', './results_xunfei/'+args.save_name+'_dev.json')
        eval(model, test_examples, alpha, beta, gama, 'dev', './results_xunfei/' + args.save_name + '_test.json')

    if args.do_lucene_predict:
        # eval(model, dev_examples, alpha, beta, gama, 'test', './results_xunfei/lucene_'+args.save_name+'_dev.json')
        eval(model, test_examples, alpha, beta, gama, 'test', './results_xunfei/lucene_' + args.save_name + '_test.json')


    if args.do_pre_finetune:
        pre_finetune_train = './data/retrieval_gold/c3-m-train-gaokao.json'
        pre_finetune_dev = './data/retrieval_gold/c3-m-test-gaokao.json'
        train_examples_pre = getSampleFeaturePreFinetune(tokenizer, args.max_seq_length, pre_finetune_train)
        dev_examples_pre = getSampleFeaturePreFinetune(tokenizer, args.max_seq_length, pre_finetune_dev)

        train_pre_finetune(model, train_examples_pre, dev_examples_pre)

    if args.do_pre_finetune_test:
        pre_finetune_test = './data/retrieval_gold/c3-m-test-gaokao.json'
        test_examples_pre = getSampleFeaturePreFinetune(tokenizer, args.max_seq_length, pre_finetune_test)

        eval_pre_finetune(model, test_examples_pre)

    if args.do_train:
        dev_examples = getSampleFeature(tokenizer, args.max_seq_length, args.dev_dataset_path, index_map, stopwords, negative_words)
        if args.dev_dataset_path2 is not None:
            dev_examples2 = getSampleFeature(tokenizer, args.max_seq_length, args.dev_dataset_path2, index_map, stopwords, negative_words)
            dev_examples += dev_examples2

        print('beging make feature')
        train_examples = getSampleFeature(tokenizer, args.max_seq_length, args.train_dataset_path,
                                          index_map, stopwords, negative_words)

        if args.train_dataset_path2 is not None:
            train_examples2 = getSampleFeature(tokenizer, args.max_seq_length, args.train_dataset_path2, index_map, stopwords, negative_words)
            train_examples += train_examples2


        # Training
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("  Num train examples = %d", len(train_examples))
            num_train_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
            logger.info("  Num train steps = %d", num_train_steps)

            # Prepare optimizer
            param_optimizer = list(model.named_parameters())

            # hack to remove pooler, which is not used
            # thus it produce None grad that break apex
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = num_train_steps
            if args.local_rank != -1:
                t_total = t_total // torch.distributed.get_world_size()
            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=args.learning_rate,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynaimc_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=args.learning_rate,
                                     warmup=args.warmup_proportion,
                                     t_total=t_total)

            global_step = 0

            if args.local_rank == -1:
                train_sampler = RandomSampler(train_examples)
            else:
                train_sampler = DistributedSampler(train_examples)

            order = list(range(len(train_examples)))
            random.shuffle(order)

            model.train()

            tr_loss, nb_tr_steps = 0, 0
            step_count = len(train_examples) // args.train_batch_size
            if step_count * args.train_batch_size < len(train_examples):
                step_count += 1
            step_trange = trange(step_count)
            right_count, count = 0, 0
            answerable_right_count, answerable_count = 0, 0
            hit, hit_count = 0, 0
            map = 0
            answer_right, answer_count = 0, 0
            for step in step_trange:
                beg_index = step * args.train_batch_size
                end_index = min((step + 1) * args.train_batch_size, len(train_examples))
                order_index = order[beg_index:end_index]
                batch = [train_examples[index] for index in order_index]

                q_ids, answers, have_negative_word, input_ids, input_mask, segment_ids, \
                orig_to_token_split_idx, orig_to_token_split, \
                scenario_range_subword, question_range_subword, option_range_subword, \
                paragraphs, scenarios, questions, options, bm25_feature, p_labels, \
                have_p_labels, word_set, word_set_idx, p_ids = get_batch_feature(batch)

                loss = model(input_ids, input_mask,
                             segment_ids, q_ids=q_ids,
                             answers=answers,
                             have_negative_word=have_negative_word,
                             orig_to_token_split_idx=orig_to_token_split_idx,
                             orig_to_token_split=orig_to_token_split,
                             word_set_idx=word_set_idx,
                             word_set=word_set,
                             paragraphs=paragraphs,
                             scenarios=scenarios,
                             questions=questions,
                             options=options,
                             scenario_range_subword=scenario_range_subword,
                             question_range_subword=question_range_subword,
                             option_range_subword=option_range_subword,
                             bm25_feature=bm25_feature,
                             p_labels=p_labels,
                             have_p_labels=have_p_labels,
                             p_ids=p_ids,
                             do_train=True,
                             do_eval=False,
                             do_pre_finetune=False,
                             retriever=args.retriever,
                             reader1=args.reader1,
                             reader2=args.reader2, p=args.p)

                tr_loss += loss.item()
                nb_tr_steps += 1

                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if ((step + 1) % args.gradient_accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_show = ' Epoch:' + str(epoch) + " loss:" + str(round(tr_loss / nb_tr_steps, 4))
                step_trange.set_postfix_str(loss_show)


            print('dev set：')
            dev_ir_hit, dev_ir_map, dev_ir_ndcg, dev_ans_accu, alpha, beta, gama = eval(model, dev_examples, None, beta,
                                                                                                gama, 'dev')

            if dev_ans_accu > dev_ans_best_accu:
                # eval(model, dev_examples, alpha, beta, gama, 'dev', args.results_save_path + args.save_name + '_dev.json')
                print('test set：')
                test_ir_hit, test_ir_map, test_ir_ndcg, test_ans_accu, alpha, beta, gama = eval(model, test_examples,
                                                                                                alpha, beta,
                                                                                                gama, 'dev',#''test',
                                                                                                args.results_save_path + args.save_name + '_test.json')
                test_ir_best_hit, test_ir_best_map, test_ir_best_ndcg = test_ir_hit, test_ir_map, test_ir_ndcg
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = '/'.join(
                    [args.output_dir, args.save_name + ".bin"])
                torch.save(model_to_save.state_dict(), output_model_file)
                dev_ans_best_accu = dev_ans_accu
                test_ans_best_accu = test_ans_accu


        print('dev result：', dev_ans_best_accu)
        print('test result：', test_ans_best_accu)
        print('ir results：', 'hit' + str(args.ir_metric_at_n) + ":", test_ir_best_hit,
              'map' + str(args.ir_metric_at_n) + ":", test_ir_best_map, 'ndcg' + str(args.ir_metric_at_n) + ":",
              test_ir_best_ndcg)
