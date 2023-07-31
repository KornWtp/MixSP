import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.moe_encoder import MixtureOfExpertsEncoder
from sentence_transformers.moe_encoder.evaluation import MoECorrelationEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import os
import gzip
import csv
from zipfile import ZipFile
import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument('--model_save_path',
					type=str,
					default='experiments/moe_cross_encoder_model',
					help='The output directory where the model checkpoints will be written.')
parser.add_argument('--model_name_or_path',
					type=str,
					default='bert-base-uncased',
					help='The model checkpoint for weights initialization.')
parser.add_argument('--batch_size', 
					type=int, 
					default=128,
					help='Batch size for training.')
parser.add_argument('--num_epochs',
					type=int,
					default=10,
					help='Total number of training epochs.')  
parser.add_argument('--num_experts',
					type=int,
					default=2,
					help='Total number of sparse expert models.')                      
parser.add_argument("--learning_rate",
					type=float,
					default=1e-4,
					help="The initial learning rate for AdamW.")     
parser.add_argument("--seed",
					type=int,
					default=1000,
					help="The random seed value")	                                   

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout



#Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)

model = MixtureOfExpertsEncoder(args.model_name_or_path, num_experts=args.num_experts, max_length=64)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = MoECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
logger.info("Start training")
start = datetime.now()

model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=args.num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          output_path=args.model_save_path,
          optimizer_params={"lr": args.learning_rate},
          use_amp=True)

stop = datetime.now()
run_time = stop - start
logger.info("Training time: " + str(run_time) + " s")

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = MixtureOfExpertsEncoder.from_pretrained(args.model_save_path)
test_evaluator = MoECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=args.model_save_path)