import random
from tensorflow import keras
import json
import numpy
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


def tokenize_stem(string_input):

    words = nltk.word_tokenize(string_input)
    stemmed_words = [stemmer.stem(w.lower()) for w in words if w != "?"]

    return stemmed_words


def load_data(data, lang):
    try:
        with open('data/'+lang+'_labels_list.json', 'r') as labels_file, open('data/'+lang+'_pattern_vocab.json', 'r') as pattern_file:
            labels_dict = json.load(labels_file)
            pattern_vocab = json.load(pattern_file)
            print('has data')
            return labels_dict, pattern_vocab
    except:
        labels_dict, pattern_vocab, input_x, input_y = generate_data(
            data, lang)
        return labels_dict, pattern_vocab


def generate_data(data, lang):
    print('does not have data')
    # Separate patterns
    questions = []
    labels = []
    labels_dict = []
    for i, intents in enumerate(data['data']):
        for pattern in intents['patterns']:
            questions.append(pattern)
            labels.append(intents['tag'])

    # Vectorize patterns (Bag of words)
    pattern_vec = CountVectorizer(binary=True, tokenizer=tokenize_stem)
    pattern_matrix = pattern_vec.fit_transform(questions)
    pattern_vocab = pattern_vec.vocabulary_
    input_x = pattern_matrix.todense()
    input_x = numpy.array(input_x)

    # Vectorize labels (Bag of words)
    labels_vec = CountVectorizer(
        binary=True, lowercase=False, token_pattern='[a-zA-Z0-9$&+,:;=?@#|<>.^*()%!-]+')
    labels_matrix = labels_vec.fit_transform(labels)
    labels_vocab = labels_vec.vocabulary_
    input_y = labels_matrix.todense()
    input_y = numpy.array(input_y)

    # Convert pattern_vocab values to integers and save it as a json file
    for key in pattern_vocab:
        pattern_vocab[key] = int(pattern_vocab[key])

    with open('data/'+lang+'_pattern_vocab.json', 'w') as file:
        json.dump(pattern_vocab, file)

    # Separate labels_vocab keys, sort them and save it to a json file
    for key in labels_vocab:
        labels_dict.append(key)
    labels_dict = sorted(labels_dict)
    # print(labels_dict)

    with open('data/'+lang+'_labels_list.json', 'w') as file:
        json.dump(labels_dict, file)

    return labels_dict, pattern_vocab, input_x, input_y


def create_model(training, output):

    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_shape=(len(training[0]),)))
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.Dense(len(output[0]), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def chat(model, labels_dict, pattern_vocab):
    print("You can chat with the bot now!")
    while True:
        inp = input("Chat: ")

        if str(inp) == '':
            print("Bot: Say something!")
        else:
            if inp.lower() == 'quit':
                break

            word_vec = CountVectorizer(
                binary=True, vocabulary=pattern_vocab, tokenizer=tokenize_stem)
            input_dict = [inp]
            matrix = word_vec.fit_transform(input_dict)

            results = model.predict([matrix.todense()])[0]
            # print(results)
            results_index = numpy.argmax(results)
            tag = labels_dict[results_index]

            if results[results_index] > 0.8:
                for tg in dataset["data"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                print("Bot: "+random.choice(responses))
            else:
                print("Sorry, I dont understand")


lang = input("\n Select language (english or albanian): ")

# Load dataset
with open('conversation_dataset/'+lang+'_dataset.json') as file:
    dataset = json.load(file)

try:
    labels_dict, pattern_vocab = load_data(dataset, lang)
    loaded_model = keras.models.load_model("keras_models/"+lang+"_model.h5")
    command = input("\nModel already exists! Do you wanna chat? (y/n): ")

    if command.lower() == 'y':
        chat(loaded_model, labels_dict, pattern_vocab)
except:
    labels_dict, pattern_vocab, input_x, input_y = generate_data(dataset, lang)
    model = create_model(input_x, input_y)
    model.fit(input_x, input_y, epochs=500, batch_size=8)
    model.save("keras_models/"+lang+"_model.h5")
    chat(model, labels_dict, pattern_vocab)
