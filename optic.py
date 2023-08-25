import string
import warnings
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from googletrans import Translator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from langdetect import detect_langs
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import ConfusionMatrixDisplay
from langdetect import detect

def create_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=STOPWORDS, min_font_size=10).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Ayarları yaparak üst ve yan boşlukları ekleyin
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)

    plt.title(title)
    plt.show()
def calculate_text_features(text):
    words = text.split()
    word_count = len(words)

    sentences = text.split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    return word_count, avg_sentence_length
def check_missing_values(data):
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_info = pd.DataFrame({'Missing Values': missing_values, 'Missing Percentage': missing_percentage})
    return missing_info

def handle_missing_values(data, threshold=10):
    columns_to_drop = data.columns[data.isnull().mean() >= threshold / 100]
    columns_to_impute = data.columns[data.isnull().mean() < threshold / 100]

    # Drop columns with high missing percentage
    data_dropped = data.drop(columns=columns_to_drop)

    # Impute missing values in remaining columns
    data_imputed = data_dropped.fillna(method='ffill')  # You can use different imputation methods

    return data_imputed

def fillBlankSpaces(file):
    for column in file.columns:
        file[column].fillna("nwg", inplace=True)
    return file

def load_data(file):
    MIN_NUMBER_TO_REMOVE = 350
    data = pd.read_excel(file)[['Customer Description', 'Subject', 'Tag: Issue Details']]
    lisOfLess = getName(data, MIN_NUMBER_TO_REMOVE)
    for i in range(len(lisOfLess)):
        data = data.drop(data[data['Tag: Issue Details'] == str(lisOfLess[i])].index)
    data = fillBlankSpaces(data)

    # Temizleme işlemi için her bir sütun için dört farklı dilde işlemi yapın
    data['Customer Description'] = clean_docs(data['Customer Description'], language="english")
    data['Customer Description'] = clean_docs(data['Customer Description'], language="turkish")
    data['Customer Description'] = clean_docs(data['Customer Description'], language="portuguese")
    data['Customer Description'] = clean_docs(data['Customer Description'], language="spanish")

    data['Subject'] = clean_docs(data['Subject'], language="english")
    data['Subject'] = clean_docs(data['Subject'], language="turkish")
    data['Subject'] = clean_docs(data['Subject'], language="portuguese")
    data['Subject'] = clean_docs(data['Subject'], language="spanish")

    return data
def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return None

def getName(data, threshold):
    fa = data['Tag: Issue Details'].value_counts()
    allList = fa.index.tolist()
    lis = []
    for i in range(fa.size):
        if fa[i] <= threshold:
            lis.append(allList[i])
    return lis


def clean_docs(docs, language):
    if language == "english":
        stops = stopwords.words("english")
    elif language == "turkish":
        stops = stopwords.words("turkish")
    elif language == "portuguese":
        stops = stopwords.words("portuguese")
    elif language == "spanish":
        stops = stopwords.words("spanish")
    else:
        raise ValueError("Invalid language selected.")

    f = open("extrastops.txt", "r")
    ex = f.read()
    extra = ex.split()
    stops = stops + extra
    final = []
    for doc in docs:
        clean_doc = removeStops(doc, stops)
        final.append(clean_doc)
    while "  " in final:
        final = final.replace("  ", " ")
    return final
def duplicate(row):
    if row['Subject'] == row['Customer Description']:
        return row['Subject']
    return row['Subject'] + ' ' + row['Customer Description']

def removeStops(text, stops):
    final = []
    text = re.sub("\(.*?\)", "()", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace(r"d{1,4}\/\d{1,4}\/\d{1,4}", "")
    text = text.replace(r"d{1,4}\.\d{1,4}\.\d{1,4}", "")
    text = text.replace(r"d{1,4}\-\d{1,4}\-\d{1,4}", "")
    text = ' '.join([i for i in text.split() if '@' not in i])
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    words = [word for word in words if word.isalpha()]
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    if final == "":
        final = 'nwg'
    return final
def main():

    data = load_data('Tagged tickets - all time - Optics -2023-07-14-09-18-45.xlsx')

    X = data[['Subject', 'Customer Description']]
    y = data['Tag: Issue Details']

    # Feature engineering
    data['Word Count'], data['Avg Sentence Length'] = zip(*data['Subject'].apply(calculate_text_features))
    data['Word Count'], data['Avg Sentence Length'] = zip(*data['Customer Description'].apply(calculate_text_features))

    # Handle missing values
    data_imputed = handle_missing_values(data)

    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(X['Subject'] + " " + X['Customer Description'])

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.1)

    model = LinearSVC(C=0.1, dual=False)

    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test, average='weighted')

    print("Accuracy statistics for the subset of data:")
    print("Accuracy rate: {:.2f}%".format(accuracy * 100))
    print("F1 score on test set: {:.2f}".format(f1))

    correct_predictions = accuracy * len(y_test)
    total_predictions = len(y_test)
    correct_ratio = correct_predictions / total_predictions
    correct_ratio_percentage = correct_ratio * 100
    correct_ratio_100_based = int(correct_ratio_percentage)

    print("Doğru tahmin/oranı: {}/{}".format(int(correct_predictions), total_predictions))

    cm = confusion_matrix(y_test, y_pred_test)
    display_labels = np.unique(y_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot(cmap='viridis', include_values=True, ax=None, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    print("\nClassification Report:")
    print("------------------------------------------------------------------------")
    print(classification_report(y_test, y_pred_test, zero_division=1))

    # Display WordCloud for Subject
    create_word_cloud(" ".join(data['Subject']), "Word Cloud for Subject")

    # Display WordCloud for Customer Description
    create_word_cloud(" ".join(data['Customer Description']), "Word Cloud for Customer Description")
if __name__ == "__main__":
    main()