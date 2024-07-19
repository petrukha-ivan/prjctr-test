import json
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from transformers import HfArgumentParser as ArgumentParser
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import root_mean_squared_error
from dataclasses import dataclass, field


@dataclass
class ModelTrainingArguments(TrainingArguments):
    
    do_train: bool = field(
        default=True, 
        metadata={'help': 'Whether to run training.'}
    )
    
    do_eval: bool = field(
        default=True, 
        metadata={'help': 'Whether to run eval on the dev set.'}
    )

    output_dir: str = field(
        default='runs',
        metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'},
    )

    data_dir: str = field(
        default='data',
        metadata={'help': 'The directory where the data in csv format stored.'},
    )

    model_checkpoint: str = field(
        default='distilbert/distilbert-base-cased',
        metadata={'help': 'Model name or path.'},
    )


def compute_rmse(eval_pred):
    predictions, labels = eval_pred
    rmse = root_mean_squared_error(labels, predictions)
    return {'rmse': rmse}


def process_dataset(samples):
    corpus = samples['excerpt']
    encoding = tokenizer(corpus, padding=True, truncation=True)
    encoding['label'] = samples['target']
    return encoding


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(ModelTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.output_dir = f'{args.output_dir}/{args.model_checkpoint}'
    args.load_best_model_at_end = True

    # Load model from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=1)

    # Describe dataset schema
    features = Features({
        'input_ids': Sequence(Value(dtype='int32')),
        'attention_mask': Sequence(Value(dtype='int32')),
        'label': Value(dtype='float32')
    })

    # Create dataset from csv
    data = pd.read_csv(f'{args.data_dir}/train.csv')
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(
        function=process_dataset,
        features=features,
        remove_columns=['id', 'url_legal', 'license', 'excerpt', 'target', 'standard_error'],
        batched=True,
        batch_size=16,
        num_proc=8
    )

    # Set format and split into train/test parts
    dataset.set_format('torch')
    dataset = dataset.train_test_split(test_size=0.3)

    # Prepare trainer
    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_rmse,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    # Train model
    if args.do_train:
        print('Training model..')
        trainer.train()
        trainer.save_model(args.output_dir + '/checkpoint-best')

    # Evaluate model
    if args.do_eval:
        print('Evaluating model..')
        metrics = trainer.evaluate()
        with open(args.output_dir + '/metrics.json', 'w') as file:
            json.dump(metrics, file, indent=4)
            print(metrics)
