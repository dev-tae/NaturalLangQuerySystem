import pandas as pd

train_df = pd.read_json('../../data/spider/train_spider.json')
dev_df = pd.read_json('../../data/spider/dev.json')

# Extracting pre-tokenized questions and queries from training DataFrame
train_question_tokens = train_df['question_toks'].tolist()
train_query_tokens = train_df['query_toks'].tolist()

# Development DataFrame
dev_question_tokens = dev_df['question_toks'].tolist()
dev_query_tokens = dev_df['query_toks'].tolist()

# Flatten the list of lists to a single list containing all tokens
all_question_tokens = [token for sublist in train_question_tokens for token in sublist]
all_query_tokens = [token for sublist in train_query_tokens for token in sublist]

# Create unique vocabularies
question_vocab = set(all_question_tokens)
query_vocab = set(all_query_tokens)

# Add special tokens
special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<OOV>']
question_vocab.update(special_tokens)
query_vocab.update(special_tokens)

# Create token-to-integer mapping
token2int_questions = {token: i for i, token in enumerate(question_vocab)}
token2int_queries = {token: i for i, token in enumerate(query_vocab)}

# Create integer-to-token mapping for later use (e.g. during inference)
int2token_questions = {i: token for token, i in token2int_questions.items()}
int2token_queries = {i: token for token, i in token2int_queries.items()}
