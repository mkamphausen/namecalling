import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import time

starttime = time.time()
print(starttime)
# Load the CSV file into a DataFrame
df = pd.read_csv('data/biden_processed/biden_filtered_english.csv',sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False, lineterminator='\n')

# ,sep=','

# Specify the column to use
column_to_use = 'tweet'  # Replace 'text_column' with the actual column name containing the text data

# Load the fine-tuned RoBERTa model and tokenizer
model_name = 'civility-lab/roberta-base-namecalling'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 32
results = []

# Function to process a batch of texts
def process_batch(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities

# Process the specified column in chunks
for i in range(0, len(df), batch_size): #replace df with df_sampled
    batch = df[column_to_use].iloc[i:i + batch_size].tolist() #replace df with df_sampled
    probabilities = process_batch(batch)
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    # Store the results
    results.extend(predicted_classes.cpu().tolist())

# Add the results as a new column in the DataFrame
df['name_calling_pred'] = results

df_output = df[['created_at', 'tweet', 'tweet_id', 'user_id', 'name_calling_pred']]

# Save the DataFrame with predictions to a new CSV file
df_output.to_csv('data/biden_processed/biden_with_predictions.csv', index=False)

endtime = time.time()
print(endtime)