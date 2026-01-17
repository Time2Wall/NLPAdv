"""
Домашнее задание 2: Дообучение энкодерных моделей
Fine-tuning BERT models for restaurant review classification
"""

import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================================
# 1. Load and preprocess data
# ============================================================================
print('\n' + '='*60)
print('Loading and preprocessing data...')
print('='*60)

data = []
with open('restaurants_reviews-327545-5892c5.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
print(f'Total samples: {len(df)}')

# Filter for ratings 1, 3, 5
df_filtered = df[df['general'].isin([1, 3, 5])].copy()
print(f'Samples after filtering (ratings 1, 3, 5): {len(df_filtered)}')

# Remap labels: 1 -> 0, 3 -> 1, 5 -> 2
label_mapping = {1: 0, 3: 1, 5: 2}
df_filtered['label'] = df_filtered['general'].map(label_mapping)

print(f'Label distribution:')
print(df_filtered['label'].value_counts().sort_index())

# Split data: train (70%), val (15%), test (15%)
RANDOM_STATE = 42

train_val_df, test_df = train_test_split(
    df_filtered,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=df_filtered['label']
)

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.15/0.85,
    random_state=RANDOM_STATE,
    stratify=train_val_df['label']
)

print(f'\nTrain size: {len(train_df)} ({len(train_df)/len(df_filtered)*100:.1f}%)')
print(f'Val size: {len(val_df)} ({len(val_df)/len(df_filtered)*100:.1f}%)')
print(f'Test size: {len(test_df)} ({len(test_df)/len(df_filtered)*100:.1f}%)')

# ============================================================================
# 2. Dataset class
# ============================================================================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# 3. Training function
# ============================================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}


def train_and_evaluate_model(
    model_name,
    train_df,
    val_df,
    test_df,
    num_labels=3,
    max_length=256,
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    patience=3
):
    print(f'\n{"="*60}')
    print(f'Training model: {model_name}')
    print(f'{"="*60}')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Create datasets
    train_dataset = ReviewDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length
    )
    val_dataset = ReviewDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length
    )
    test_dataset = ReviewDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer,
        max_length
    )

    # Output directory
    output_dir = f'./results/{model_name.replace("/", "_")}'

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        save_total_limit=2,
        report_to='none',
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
    )

    # Training
    start_time = time.time()
    train_result = trainer.train()
    total_training_time = time.time() - start_time

    # Get training info
    train_logs = trainer.state.log_history

    # Find epoch with minimum validation loss
    eval_losses = [(i, log['eval_loss']) for i, log in enumerate(train_logs) if 'eval_loss' in log]
    if eval_losses:
        min_loss_entry = min(eval_losses, key=lambda x: x[1])
        best_epoch = [log['epoch'] for log in train_logs if 'eval_loss' in log][eval_losses.index(min_loss_entry)]
    else:
        best_epoch = num_epochs

    # Calculate time per iteration
    total_steps = train_result.global_step
    time_per_iteration = total_training_time / total_steps if total_steps > 0 else 0

    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    test_accuracy = test_results.get('eval_accuracy', 0)

    print(f'\nResults for {model_name}:')
    print(f'  Best epoch (min val loss): {best_epoch}')
    print(f'  Time per iteration: {time_per_iteration:.3f}s')
    print(f'  Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)')
    print(f'  Test accuracy: {test_accuracy:.4f}')

    # Cleanup to free memory
    del model
    del trainer
    torch.cuda.empty_cache()

    return {
        'model_name': model_name,
        'best_epoch': best_epoch,
        'time_per_iteration': time_per_iteration,
        'total_training_time': total_training_time,
        'test_accuracy': test_accuracy
    }

# ============================================================================
# 4. Train all models
# ============================================================================
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
PATIENCE = 3

results = []

# Model 1: ruBert-base
result = train_and_evaluate_model(
    model_name='sberbank-ai/ruBert-base',
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE
)
results.append(result)

# Model 2: rubert-tiny2
result = train_and_evaluate_model(
    model_name='cointegrated/rubert-tiny2',
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE * 2,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE
)
results.append(result)

# Model 3: bert-base-multilingual-cased
result = train_and_evaluate_model(
    model_name='google-bert/bert-base-multilingual-cased',
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE
)
results.append(result)

# ============================================================================
# 5. Results summary
# ============================================================================
print('\n' + '='*80)
print('RESULTS SUMMARY')
print('='*80)

results_df = pd.DataFrame([
    {
        'Model': r['model_name'],
        'Best Epoch': r['best_epoch'],
        'Time/Iter (s)': round(r['time_per_iteration'], 3),
        'Total Time (min)': round(r['total_training_time'] / 60, 2),
        'Test Accuracy': round(r['test_accuracy'], 4)
    }
    for r in results
])

print(results_df.to_string(index=False))

# Save results to file
results_df.to_csv('training_results.csv', index=False)
print('\nResults saved to training_results.csv')

print('\n' + '='*80)
print('TRAINING COMPLETE!')
print('='*80)
