import torch

filepath = "wonderland.txt"
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
characters = sorted(set(text))
vocab_size = len(characters)

#Charecter level tokenizer
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

# print(characters)
# print(vocab_size)
# print(char_to_index['a'])
encoded_text = []
for c in text:
    encoded_text.append(char_to_index[c])
# print(encoded_text)
data = torch.tensor(encoded_text, dtype = torch.long)
# print(data[:1000])

##split into train-test split 90-10
n = int(0.9*len(data))
train_data = data[n:]
test_data = data[n:]



##
block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input in {context} the ta4rget: {target}")