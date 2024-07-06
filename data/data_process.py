from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import pandas as pd
import openpyxl
import os

# dataset path
DATASET_PATH = "../dataset/한국어-영어 번역 말뭉치(기술과학)/"
TRAINING_DATASET_PATH = DATASET_PATH + "Training/"

data_a = pd.read_csv(TRAINING_DATASET_PATH + '1113_tech_train_set_1195228.csv', encoding='utf-8')

ko_data = data_a['ko']
en_data = data_a['en']

# Create tokenizer training data

if not os.path.exists(os.path.join(TRAINING_DATASET_PATH, 'train_korean.txt')):
    with open(os.path.join(TRAINING_DATASET_PATH, 'train_korean.txt'), 'w', encoding='utf-8') as f:
        for ko in ko_data:
            print(ko, file=f)

if not os.path.exists(os.path.join(TRAINING_DATASET_PATH, 'train_english.txt')):
    with open(os.path.join(TRAINING_DATASET_PATH, 'train_english.txt'), 'w', encoding='utf-8') as f:
        for en in en_data:
            print(en, file=f)

# tokenizer training
print("Tokenizer training start")

params = {
    'batch_size': 64,
    'num_epoch' : 50,
    'dropout': 0.1,
    'min_frequency': 3,

    'vocab_size': 20000,
    'num_layers': 6, # transformer number of num_layers
    'num_heads': 8, # number of attention heads in the multi-head attention models
    'hidden_dim': 512, # hidden dimension of model
    'ffn_din': 2048, # feed forward network dimension
}

tokenizer_model = models.BPE()

ko_tokenizer = Tokenizer(tokenizer_model)
en_tokenizer = Tokenizer(tokenizer_model)

trainer = trainers.BpeTrainer(
    special_tokens=['[PAD]', '[SOS]', '[EOS]', '[UNK]'],
    vocab_size=params['vocab_size'],
    min_frequency=params['min_frequency'],
    suffix=''
)

ko_tokenizer.train(files=[os.path.join(TRAINING_DATASET_PATH, 'train_korean.txt')], trainer=trainer)
en_tokenizer.train(files=[os.path.join(TRAINING_DATASET_PATH, 'train_english.txt')], trainer=trainer)

pad_idx = ko_tokenizer.token_to_id('[PAD]')
sos_idx = ko_tokenizer.token_to_id('[SOS]')
eos_idx = ko_tokenizer.token_to_id('[EOS]')

# tokenizer check
print("Tokenizer Check")
ko_encoded_data = ko_tokenizer.encode_batch(ko_data)
en_encoded_data = en_tokenizer.encode_batch(en_data)

for origin, processed in zip(ko_data[:3], ko_encoded_data[:3]):
    print(f'origin: {origin}')
    print(f'processed: {processed.tokens} \n')


