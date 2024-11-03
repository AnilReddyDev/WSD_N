from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the model and tokenizer
model_name = "mwesner/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Load your dataset
data = load_dataset("conceptnet5/conceptnet5", "conceptnet5", split="train")

# Convert the dataset to a Pandas DataFrame
data_df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
data_df['label'] = label_encoder.fit_transform(data_df['label'])

# Reduce dataset size to a manageable amount (e.g., 1000 samples)
data_df = data_df.sample(n=1000, random_state=42)

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=42)  # 20% for validation

# Convert back to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

# Preprocess dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Set up training arguments and trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce batch size
    per_device_eval_batch_size=4,  # Reduce batch size
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=50,  # Log every 50 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train and evaluate
trainer.train()
trainer.evaluate()
