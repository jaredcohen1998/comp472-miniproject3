import gensim.downloader as api
from gensim.similarities import MatrixSimilarity
import pandas as pd
import random

def experiment_with_model(modelName, synonymFile):
    targetFile = F"data/{modelName}-model.csv"

    print(F"\n\n===LOADING MODEL {modelName}===")
    model = api.load(modelName)

    print(F"Loading {synonymFile}...")

    question_word_list = []
    answer_word_list = []
    guess_word_list = []
    label_list = []
    num_words = 0
    num_correct = 0
    num_guesses = 0
    csv = pd.read_csv(synonymFile)

    print(F"Gathering data to write to {targetFile}...")

    for row in csv.iterrows():
        r = row[1]
        question_word_list.append(r['question'])
        answer_word_list.append(r['answer'])

        cs_list = []
        keyword_exists = r['question'] in model.key_to_index
        w1_exists = r['0'] in model.key_to_index
        w2_exists = r['1'] in model.key_to_index
        w3_exists = r['2'] in model.key_to_index
        w4_exists = r['3'] in model.key_to_index
        words_exists = w1_exists == True or w2_exists == True or w3_exists == True or w4_exists == True

        if (keyword_exists == True):
            if (w1_exists == True):
                cs_list.append(model.similarity(r['question'], r['0']))
            if (w2_exists == True):
                cs_list.append(model.similarity(r['question'], r['1']))
            if (w3_exists == True):
                cs_list.append(model.similarity(r['question'], r['2']))
            if (w4_exists == True):
                cs_list.append(model.similarity(r['question'], r['3']))

        num_words = num_words + 1
        if (keyword_exists == False or words_exists == False):
            label_list.append("guess")
            guess_word = random.choice([r['0'], r['1'], r['2'], r['3']])
            guess_word_list.append(guess_word)
            num_guesses = num_guesses + 1
        else:
            cs_max = max(cs_list)
            cs_max_index = cs_list.index(cs_max)

            guess_word = r[str(cs_max_index)]
            guess_word_list.append(guess_word)

            if (guess_word == r['answer']):
                label_list.append("correct")
                num_correct = num_correct + 1
            else:
                label_list.append("wrong")

    df = pd.DataFrame({"question": question_word_list, "answer": answer_word_list, "system guess word": guess_word_list, "label": label_list})
    df.to_csv(targetFile, index=False)

    print(F"Wrote to file {targetFile}")

    if (num_words - num_guesses == 0):
        accuracy = 0
    else:
        accuracy = num_correct / (num_words - num_guesses)

    return (modelName, model.vectors.shape[0], num_correct, num_words, num_guesses, accuracy)


def main():
    synonymFile = "data/synonyms.csv"
    models_list = [
        "word2vec-google-news-300", "glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-50", "glove-twitter-100"
    ]

    modelName_list = []
    model_size_list = []
    num_correct_list = []
    num_answered_list = []
    accuracy_list = []

    for i in range(len(models_list)):
        model_name, model_size, num_correct, num_words, num_guesses, accuracy = experiment_with_model(models_list[i], synonymFile)

        modelName_list.append(model_name)
        model_size_list.append(model_size)
        num_correct_list.append(num_correct)
        num_answered_list.append(num_words - num_guesses)
        if (num_words - num_guesses != 0):
            accuracy_list.append(num_correct / (num_words - num_guesses))
        else:
            accuracy_list.append(0)

    targetFile = "data/analysis.csv"
    print(F"\nWriting to file {targetFile}...")

    df = pd.DataFrame({"model name": modelName_list, "size": model_size_list, "correct labels": num_correct_list, "answered": num_answered_list, "accuracy": accuracy_list})
    df.to_csv(targetFile, index=False)

    print(F"Wrote to file {targetFile}")

if __name__ == "__main__":
    main()