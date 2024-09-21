import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report
import numpy as np


# Load the two datasets

# df1 = pd.read_csv('./data/US Election 2020 Tweets/hashtag_joebiden.csv', sep=',', chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')
# df2 = pd.read_csv('./data/US Election 2020 Tweets/hashtag_donaldtrump.csv', sep=',', chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')

# CHUNK IT DOWN

chunk_size = 10000
chunks = pd.read_csv('./data/US Election 2020 Tweets/hashtag_joebiden.csv', sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')
# for chunk in chunks:
#     # Process each chunk
#     print(chunk.head())

# Combine the two datasets
# df = pd.concat([df1, df2], ignore_index=True)
# df = chunks

# Preprocess the combined dataset
df = chunks.dropna(subset=['tweet'])  # Ensure there are no missing values in the 'tweet' column

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the custom Dataset class (same as before)
class TweetDataset(Dataset):
    def __init__(self, tweets, labels, tokenizer, max_len):
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.tweets)
    
    def __getitem__(self, index):
        tweet = self.tweets.iloc[index]
        label = self.labels.iloc[index]

        encoding = self.tokenizer.encode_plus(
            tweet,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'tweet_text': tweet,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Function to create data loaders
def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = TweetDataset(
        tweets=df['tweet'],
        labels=df['label'],  # Adjust column name as necessary
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4
    )

# Load pre-trained roBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2,  # Adjust for your specific case
    output_attentions=False,
    output_hidden_states=False
)

# Create data loaders
train_data_loader = create_data_loader(train_df, tokenizer, max_len=128, batch_size=16)
val_data_loader = create_data_loader(val_df, tokenizer, max_len=128, batch_size=16)

# Set up training components
EPOCHS = 4
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds)
            true_labels.extend(labels)

    predictions = torch.stack(predictions).cpu()
    true_labels = torch.stack(true_labels).cpu()

    print(classification_report(true_labels, predictions, target_names=["Not Name-Calling", "Name-Calling"]))

# Training loop
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        optimizer,
        device,
        scheduler
    )
    print(f'Train loss {train_loss}, accuracy {train_acc}')

# Evaluate the model
eval_model(model, val_data_loader, device)

# Save the trained model
model.save_pretrained('namecalling_roberta')
tokenizer.save_pretrained('namecalling_roberta')