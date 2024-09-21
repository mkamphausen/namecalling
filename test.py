from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('data/US Election 2020 Tweets/hashtag_joebiden.csv',sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')

# Preview the DataFrame
print(df.head())

small_df = df.sample(n=20000, random_state=42)
print(small_df.head())

# Load pre-trained model and tokenizer
model_name = 'civility-lab/roberta-base-namecalling'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
 
batch_size = 32
results = []

# Function to process a batch of texts
def process_batch(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities

# Process the specified column in chunks
for i in range(0, len(small_df), batch_size):
    batch = small_df['tweet'].iloc[i:i + batch_size].tolist()
    probabilities = process_batch(batch)
    predicted_classes = torch.argmax(probabilities, dim=1)
    results.extend(predicted_classes.tolist())

small_df['name_calling_pred'] = results
small_df.to_csv('output_with_predictions.csv', index=False)