import gensim.downloader as api
from gensim.similarities import MatrixSimilarity
import pandas as pd
import random

def taskone():
    modelName = "word2vec-google-news-300"
    targetFile = F"{modelName}-model.csv"
    synonymFile = "data/synonyms.csv"

    print(F"\nLoading model {modelName}...")
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
    
    targetFile = "analysis.csv"
    print(F"\nWriting to file {targetFile}...")

    modelName_list = []
    model_size_list = []
    num_correct_list = []
    num_answered_list = []
    accuracy_list = []

    modelName_list.append(modelName)
    model_size_list.append(model.vectors.shape[0])
    num_correct_list.append(num_correct)
    num_answered_list.append(num_words - num_guesses)
    if (num_words - num_guesses != 0):
        accuracy_list.append(num_correct / (num_words - num_guesses))
    else:
        accuracy_list.append(0)

    df = pd.DataFrame({"model name": modelName_list, "size": model_size_list, "correct labels": num_correct_list, "answered": num_answered_list, "accuracy": accuracy_list})
    df.to_csv(targetFile, index=False)

    print(F"Wrote to file {targetFile}")


def main():
    taskone()

if __name__ == "__main__":
    main()