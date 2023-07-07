from torch.utils.data import DataLoader
import math
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

#Configuration
train_batch_size = 128
num_epochs = 10
model_save_path = 'experiments/moe_model_num_experts_5'


model = MixtureOfExpertsEncoder('bert-base-uncased', num_experts=5)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# We add an evaluator, which evaluates the performance during training
evaluator = MoECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={"lr": 1e-4},
          use_amp=True)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = MixtureOfExpertsEncoder.from_pretrained(model_save_path)
test_evaluator = MoECorrelationEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)