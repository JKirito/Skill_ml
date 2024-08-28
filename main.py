import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load the CSV data
df = pd.read_csv('skills.csv')

# Preprocess the data
def preprocess_data(df):
    # Combine relevant columns into a single text column
    df['text'] = df['Technical Skills'] + " " + df['Soft Skills'] + " " + df['Experience']
    # Create labels (for simplicity, let's assume a binary classification with dummy labels)
    df['label'] = 0  # Replace with actual labels if available
    return df[['text', 'label']]

df = preprocess_data(df)


# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)
print(dataset)

# # Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2)
print(train_test_split)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# # Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

print(tokenized_train_dataset)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    evaluation_strategy="epoch",     # evaluation strategy to adopt during training
    learning_rate=2e-5,              # learning rate
    per_device_train_batch_size=4,  # Smaller batch size
    gradient_accumulation_steps=2,   # Accumulate gradients over 2 steps
    per_device_eval_batch_size=8,   # batch size for evaluation
    num_train_epochs=3,              # total number of training epochs
    weight_decay=0.01,               # strength of weight decay
)

trainer = Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    train_dataset=tokenized_train_dataset,  # training dataset
    eval_dataset=tokenized_test_dataset     # evaluation dataset
)

trainer.train()  # Start training

# Generate predictions on the test dataset
predictions = trainer.predict(tokenized_test_dataset)

# Extract the predicted labels
predicted_labels = predictions.predictions.argmax(axis=1)

# Add predictions to the test dataset
test_dataset = test_dataset.add_column("predicted_label", predicted_labels)

# Print the predictions
print(test_dataset)



# # Example usage
# def tokenize_skills(skills):
#     return tokenizer(skills, padding=True, truncation=True, max_length=128, return_tensors="pt")

# candidate_skills = ["Python programming", "Data analysis", "Machine learning"]
# tokenized_skills = tokenize_skills(candidate_skills)

# job_requirements = ["Experience in Python", "Knowledge of machine learning algorithms"]
# tokenized_requirements = tokenize_skills(job_requirements)

# print(tokenized_skills)
