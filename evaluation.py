import sys
import io, os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import transformers
from sentence_transformers.mixsp_encoder import MixSPEncoder
from sentence_transformers.mixsp_encoder.evaluation import MixSPCorrelationEvaluator, MixSPCorrelationEvaluatorAUC
from data import load_data

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")
    parser.add_argument("--task_set", type=str, 
            choices=["sts", "7_standard_sts", "binary_classification", "na"],
            default="sts",
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")
 
    args = parser.parse_args()
    print(args)
    
    ### read datasets
    if args.task_set == "sts":
        all_pairs, all_test, dev_samples = load_data(args.task_set)
    elif args.task_set == "7_standard_sts":
        all_pairs, all_test, dev_samples = load_data(args.task_set)
    elif args.task_set == "binary_classification":
        all_pairs = []
        all_test = []
        dev_samples = []
        for name in ["qqp", "qnli", "mrpc"]:
            pairs, test, dev = load_data(name)
            all_pairs.append(pairs)
            all_test.append(test)
            dev_samples.append(dev)
    else:
        raise NotImplementedError

    # Load moe sentence transformers' model checkpoint
    model = MixSPEncoder.from_pretrained(args.model_name_or_path)

    if args.task_set == "sts":
        all_sts_name = {"bio": "BIOSSES",
                "cdsc_r_val": "CDSC-R (Val)",
                "cdsc_r_test": "CDSC-R (Test)"}
        
        task_names = []
        scores = []
        logging.info ("########### Main STS Results ##########")
        for name, data in all_test.items():
            test_evaluator = MixSPCorrelationEvaluator.from_input_examples(data, name=name)
            task_names.append(all_sts_name[name])
            scores.append("%.2f" % (test_evaluator(model) * 100))
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
    elif args.task_set == "7_standard_sts":
        all_sts_name = {"2012": "STS12",
                "2013": "STS13",
                "2014": "STS14",
                "2015": "STS15",
                "2016": "STS16",
                "stsb": "STSBenchmark",
                "sickr": "SICKRelatedness"}
    
        task_names = []
        scores = []
        logging.info ("########### 7 Standard STS Results ##########")
        for name, data in all_test.items():
            test_evaluator = MixSPCorrelationEvaluator.from_input_examples(data, name=name)
            task_names.append(all_sts_name[name])
            scores.append("%.2f" % (test_evaluator(model) * 100))
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
    elif args.task_set == "binary_classification":
        transfer_name = {"qqp": "QQP",
                "qnli": "QNLI",
                "mrpc": "MRPC"}

        task_names = []
        scores = []
        logging.info ("########### Binary Text Classification Results ###########")
        for test in all_test:
            for name, data in test.items():
                test_evaluator = MixSPCorrelationEvaluatorAUC.from_input_examples(data, name=name)
                task_names.append(transfer_name[name])
                scores.append("%.2f" % (test_evaluator(model) * 100))
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()