# encoding=utf-8

"""
Evaluate results and calculate metrics

Usage:
    eval.py [options] TEST_SET RESULT_FILE

Options:
    -h --help                   show this screen.
    --metrics=<arg...>          metrics to calculate [default: accuracy,recall,distance,nlg]
    --eval-class=<str>          the class used to evaluate [default: Evaluator]
"""
import json
import logging
import stanfordnlp
from utils.common import *
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple
from utils.common import word_level_edit_distance
from nlgeval import NLGEval
from collections import OrderedDict

from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
                 'plbart': (PLBartConfig, PLBartForConditionalGeneration, PLBartTokenizer)}

logging.basicConfig(level=logging.INFO)
EMPTY_TOKEN = '<empty>'

import argparse

class BaseMetric(ABC):
    @abstractmethod
    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> float:
        """
        :param hypos: each hypo contains k sents, for accuracy, only use the first sent, for recall, use k sents
        :param references: the dst desc sents
        :param src_references: the src desc sents
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def is_equal(hypo: List[str], ref: List[str]):
        if hypo == ref:
            return True
        if ref[-1] in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~_'.split() and ref[:-1] == hypo:
            return True
        return False


class Accuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        super(Accuracy, self).__init__()
        self.correct_count = 0

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]] = None, *args, **kwargs) -> dict:
        total = 0
        correct = 0
        for hypo_list, ref in zip(hypos, references):
            hypo = hypo_list[0]
            if not hypo:
                hypo = [EMPTY_TOKEN]
            assert (type(hypo[0]) == str)
            assert (type(ref[0]) == str)
            total += 1
            if self.is_equal(hypo, ref):
                correct += 1
        return {'accuracy': correct / total, 'correct_count': correct}


class Recall(BaseMetric):
    def __init__(self, k: int = 5, *args, **kwargs):
        super(Recall, self).__init__()
        self.k = k

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]] = None, *args, **kwargs) -> float:
        total = 0
        correct = 0
        for hypo_list, ref in zip(hypos, references):
            total += 1
            for hypo in hypo_list:
                if self.is_equal(hypo, ref):
                    correct += 1
                    break
        return correct / total


class EditDistance(BaseMetric):
    def __init__(self, *args, **kwargs):
        super(EditDistance, self).__init__()

    @staticmethod
    def edit_distance(sent1: List[str], sent2: List[str]) -> int:
        return word_level_edit_distance(sent1, sent2)

    @classmethod
    def relative_distance(cls, src_ref_dis, hypo_ref_dis):
        if src_ref_dis == 0:
            logging.error("src_ref is the same as ref.")
            src_ref_dis = 1
        return hypo_ref_dis / src_ref_dis

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> dict:
        src_distances = []
        hypo_distances = []
        rel_distances = []
        for idx, (hypo_list, ref, src_ref) in enumerate(zip(hypos, references, src_references)):
            hypo = hypo_list[0]
            hypo_ref_dis = self.edit_distance(hypo, ref)
            src_ref_dis = self.edit_distance(src_ref, ref)
            src_distances.append(src_ref_dis)
            hypo_distances.append(hypo_ref_dis)
            rel_distances.append(self.relative_distance(src_ref_dis, hypo_ref_dis))
        rel_dis = float(np.mean(rel_distances))
        src_dis = float(np.mean(src_distances))
        hypo_dis = float(np.mean(hypo_distances))
        # return float(np.mean(distances))
        return {"rel_distance": rel_dis, "hypo_distance": hypo_dis, "src_distance": src_dis}


class NLGMetrics(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)

    @staticmethod
    def prepare_sent(tokens: List[str]) -> str:
        return recover_desc(tokens)

    def eval(self, hypos: Iterable[List[List[str]]], references: Iterable[List[str]],
             src_references: Iterable[List[str]], *args, **kwargs) -> dict:
        # List[str]
        first_hypos = [self.prepare_sent(hypo_list[0]) for hypo_list in hypos]
        src_ref_strs = [self.prepare_sent(src_ref) for src_ref in src_references]
        # List[List[str]]
        references_lists = [[self.prepare_sent(ref) for ref in references]]
        # distinct
        metrics_dict = self.nlgeval.compute_metrics(references_lists, first_hypos)
        # relative improve
        src_metrics_dict = self.nlgeval.compute_metrics(references_lists, src_ref_strs)
        relative_metrics_dict = OrderedDict({})
        for key in metrics_dict:
            relative_metrics_dict[key] = (metrics_dict[key] - src_metrics_dict[key]) / src_metrics_dict[key]
        return {
            'Bleu_4': metrics_dict['Bleu_4'],
            'METEOR': metrics_dict['METEOR']
        }


class StanfordNLPTool:
    def __init__(self):
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma', lang='en')

    def lemmatize(self, sent: List[str]):
        doc = self.nlp(" ".join(sent))
        return [w.lemma for s in doc.sentences for w in s.words]


class BaseEvaluator(ABC):
    @abstractmethod
    def load_hypos_and_refs(self) -> Tuple[List[List[List[str]]], List[List[str]], List[List[str]]]:
        pass


class Evaluator(BaseEvaluator):
    METRIC_MAP = {
        "accuracy": Accuracy(),
        "recall": Recall(k=5),
        "distance": EditDistance(),
        "nlg": NLGMetrics()
    }

    def __init__(self, args: dict, metric_map: dict = None, no_lemma: bool = True):
        self.args = args
        self.metric_map = metric_map if metric_map else self.METRIC_MAP
        self.no_lemma = no_lemma
        self.nlp = StanfordNLPTool() if not no_lemma else None
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.tokenizer = tokenizer_class.from_pretrained('Salesforce/codet5-base')

    def load_hypos(self):
        with open(self.args.hyps, 'r') as f:
            results = f.readlines()
        return [i.replace("\n","").strip() for i in results]

    @staticmethod
    def normalize_hypos(hypos, src_references):
        new_hypos = []
        for hypo_list, src_sent in zip(hypos, src_references):
            if not hypo_list:
                print("find empty hypo list")
                hypo_list = src_sent
            new_hypos.append(hypo_list)
        return new_hypos

    def load_hypos_and_refs(self, args):
        refs = []
        src_refs = []
        with open(args.test_set,'r') as f:
            test = f.readlines()
            for i in test:
                i = json.loads(i)
                src_refs.append(i['old_comment'])
                refs.append(i['new_comment'])
        references = [self.tokenizer.tokenize(i) for i in refs]
        src_references = [self.tokenizer.tokenize(i) for i in src_refs]
        hypos = self.load_hypos()
        hypos = self.normalize_hypos(hypos, src_references)
        hypos = [[self.tokenizer.tokenize(i)] for i in hypos]
        return hypos, references, src_references

    def cal_metrics(self, metrics: Iterable[str], hypos: List[List[List[str]]], references: List[List[str]],
                    src_references: List[List[str]]):
        results = {}
        for metric in metrics:
            instance = self.metric_map[metric.lower()]
            results[metric] = instance.eval(hypos, references, src_references)
        return results

    def evaluate(self, args):
        metrics = args.metrics.split(',')
        hypos, references, src_references = self.load_hypos_and_refs(args)
        print("==============",hypos[0])
        print("==============",references[0])
        print("==============",src_references[0])
        assert type(hypos[0][0]) == type(references[0])
        results = self.cal_metrics(metrics, hypos, references, src_references)
        logging.info(results)
        print(results)
        return results


def evaluate(args, no_lemma=True):
    evaluator = Evaluator(args, no_lemma=no_lemma)
    res =  evaluator.evaluate(args)
    # p = args['RESULT_FILE'][:args['RESULT_FILE'].index('/')+1]
    # with open(f"{p}res.txt", 'w') as f:
    #     f.write(str(res))
    # print("store path: ",p)
    # print("test result", res)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str)
    parser.add_argument("--hyps", type=str)
    parser.add_argument("--model_type", type=str,  default='codet5')
    parser.add_argument("--metrics", type=str, default='accuracy,recall,distance,nlg')
    args = parser.parse_args()
    print("args: ",args)
    evaluate(args, True)
