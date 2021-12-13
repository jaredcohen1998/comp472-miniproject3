import pandas
import matplotlib.pyplot as plt

humanAccuracy = 0.8557
humanAnswered = 80
randomAccuracy = 0.25
results = pandas.read_csv('data//analysis.csv')
df = pandas.DataFrame(results)
# Names in file are to big for a nice bar chart.
#Models = list(df.iloc[:,0])
Models = ["GoogleNews", "WikiGiga", "WikiNews", "Twitter50", "Twitter100"]
correctAnswers = list(df.iloc[:, 2])
modelAnswered = list(df.iloc[:, 3])
modelAccuracies = list(df.iloc[:, 4])

print("Generating Correct Answers chart...")
plt.bar(Models, correctAnswers)
plt.title("Number of Correct Answers by Models")
plt.xlabel("Model")
plt.ylabel("Correct Answers")
plt.savefig('results//CorrectAnswers.png', dpi=400)
plt.show()
print("Chart complete!")

print("Generating Attempted Answers chart...")
plt.bar(Models, modelAnswered)
plt.title("Number of Attempted Answers by Models")
plt.xlabel("Model")
plt.ylabel("Attempted Answers")
plt.ylim(75)
plt.savefig('results//AttemptedAnswers.png', dpi=400)
plt.show()
print("Chart complete!")

print("Generating Accuracy chart...")
plt.bar(Models, modelAccuracies)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy of Models")
plt.ylim(.40)
plt.savefig('results//ModelAccuracies.png', dpi=400)
plt.show()
print("Chart complete!")

Models.append("Human")
Models.append("Random")
modelAccuracies.append(humanAccuracy)
modelAccuracies.append(0.2)

print("Generating Appended Accuracy chart...")
plt.bar(Models, modelAccuracies)
plt.title("Accuracy of Models Compared to Human Standard and Random Chance")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.savefig('results//AccuracyComparisons.png', dpi=400)
plt.show()
print("Chart complete!")
