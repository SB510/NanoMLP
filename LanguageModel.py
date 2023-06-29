from NN import *
"""Using Alice in Wonderland As training text"""
filepath = "wonderland.txt"
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
characters = sorted(set(text))
vocab_size = len(characters)

#Charecter level tokenizer
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

encoded_text = []
for c in text:
    encoded_text.append(char_to_index[c])
data = encoded_text

##split into train-test split 90-10
n = int(0.9*len(data))
train_data = data[n:]
test_data = data[n:]


##Training with JUST input context of 8 
block_size = 8
xtrain = []
ytrain = []
for i in range(block_size, len(train_data)-1, 1):
    xtrain.append(train_data[i-block_size:i])
    ytrain.append(train_data[i])
print(xtrain[:100])
print(ytrain[:100])

mlp = MLP(8, [4,4,1])
network = NN(mlp, xtrain[:100], ytrain[:100])
network.train(200, 0.005)
phrase = "I am god"
print(phrase)
encoded_phrase = []
for c in text:
    encoded_phrase.append(char_to_index[c])
for i in range(15):
    prediction = network.predict(encoded_phrase)
    print(prediction)
    phrase += index_to_char[int(prediction.data)]
    encoded_phrase.pop(0)
    encoded_phrase.append(prediction)
print(phrase)







##frame for multiple contexts in 1 data point
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input in {context} the ta4rget: {target}")


