import os
import pickle
import string
import time
from datetime import datetime

import nltk
import pandas as pd
import seaborn as sns
from nltk import SnowballStemmer
# from keras.src.metrics import F1Score
from nltk.corpus import stopwords  # библиотека для стоп-слов
import pymorphy2  # лемматизация слов
# from tensorflow.python.keras.metrics import Recall, Precision
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Embedding, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K

morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words("russian"))
# nltk.download('snowball_data')
stemmer = SnowballStemmer("russian")


def remove_digits_and_punctuation(text):
    text = text.lower()  # нижний регистр
    punctuation = string.punctuation + '«»'  # Создаем строку, содержащую все знаки препинания
    result = ''.join(' ' if char in punctuation or char.isdigit() else char for char in
                     text)  # Удаляем все знаки препинания и цифры из текста, заменяя их на пробелы
    result = ' '.join(result.split())  # Удаляем лишние пробелы с помощью split и join
    result = result.replace('ё', 'е')  # заменяем букву "ё" на букву "е"     result = result.replace('€', 'деньги')
    return result


def lemmatize(text):
    words = text.split()
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    return lemmatized_text.replace('ё', 'е')


def stem(text):
    # Разбиваем текст на отдельные слова
    words = text.split()
    # Стемминг каждого слова
    stemmed_words = [stemmer.stem(word) for word in words]
    # Склеиваем стеммированные слова обратно в текст
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text.replace('ё', 'е')


# Импортируем список стоп-слов на русском языке из библиотеки NLTK
stop_words = set(stopwords.words("russian"))

# Функция для удаления стоп-слов из текста.
def remove_stop_words(text):
    words = text.split()  # Разбиваем текст на слова
    word_to_keep = "не"  # Слово, которое нужно сохранить
    filtered_words = [word for word in words
                      if word.lower() not in stop_words or word.lower() == word_to_keep.lower()]
    # Фильтруем слова, оставляя только те, которые не являются стоп-словами, за исключением слова "не"
    return ' '.join(filtered_words)  # Объединяем слова обратно в текст


# путь к файлу Excel
excel_file_path = "C:\\User\\user\\PycharmProjects\\sentimenLSTM\\12345.xlsx"
df = pd.read_excel(excel_file_path)

def filter_and_limit_data(df, column_index, value, start_index, end_index):
    if value == 1:
        filter_mask = ((df.iloc[:, column_index] == value) | (df.iloc[:, column_index] == 2))
        if column_index != 0:
            filter_mask &= ((df.iloc[:, 0] != 4) & (df.iloc[:, 0] != 5))
    elif value == 5:
        filter_mask = (df.iloc[:, column_index] == value)
    else:
        filter_mask = (df.iloc[:, column_index] == value)
    for col_index, col in enumerate(df.columns):
        if col_index != column_index and col_index != 0:
            if value == 1:
                filter_mask &= (df[col] != 4) & (df[col] != 5)
            elif value == 5:
                filter_mask &= (df[col] != 1) & (df[col] != 2) & (df[col] != 3) & (df[col] != 4)
            else:
                filter_mask &= (df[col] != 4) & (df[col] != 5)
    filtered_data = df[filter_mask]
    filtered_data = filtered_data.iloc[:, 6]
    filtered_data = filtered_data.tolist()
    filtered_data = filtered_data[start_index:end_index]
    return filtered_data
#################################################
#################################################
#################################################
#################################################

save = 1 # 0 - ничего 1 для сохранения
epoch = 5
print("Получает данные из бд ")

db_num = 2
if db_num==100:
    db_num_valid_s = 30
    db_num_valid_e = 30
elif db_num == 270:
    db_num_valid_s = 50
    db_num_valid_e = 50
elif db_num==20:
    db_num_valid_s = 5
    db_num_valid_e = 5
elif db_num==10:
    db_num_valid_s = 2
    db_num_valid_e = 2

#################################################
#################################################
#################################################
#################################################

# Получаем первые 300 элементов для каждой переменной
price_data_false = filter_and_limit_data(df, 1, 1, 0, db_num)
sleep_data_false = filter_and_limit_data(df, 2, 1, 0, db_num)
numberH_data_false = filter_and_limit_data(df, 3, 1, 0, db_num)
clean_data_false = filter_and_limit_data(df, 4, 1, 0, db_num)
service_data_false = filter_and_limit_data(df, 5, 1, 0, db_num)
# for item in service_data_false:
#     print(item)
# print("Количество строк в списке price_data_true: ", len(service_data_false))
# raise Exception("Прерывание выполнения кода")

# списка для каждой переменной, начиная с 301 элемента
price_data_valid_false = filter_and_limit_data(df, 1, 1, db_num_valid_s,  db_num_valid_e)
sleep_data_valid_false = filter_and_limit_data(df, 2, 1, db_num_valid_s,  db_num_valid_e)
numberH_data_valid_false = filter_and_limit_data(df, 3, 1, db_num_valid_s,  db_num_valid_e)
clean_data_valid_false = filter_and_limit_data(df, 4, 1, db_num_valid_s, db_num_valid_e)
service_data_valid_false = filter_and_limit_data(df, 5, 1, db_num_valid_s,  db_num_valid_e)
price_data_true = filter_and_limit_data(df, 1, 5, 0, db_num)

sleep_data_true = filter_and_limit_data(df, 2, 5, 0, db_num)
numberH_data_true = filter_and_limit_data(df, 3, 5, 0, db_num)
clean_data_true = filter_and_limit_data(df, 4, 5, 0, db_num)
service_data_true = filter_and_limit_data(df, 5, 5, 0, db_num)
price_data_valid_true = filter_and_limit_data(df, 1, 5, db_num_valid_s, db_num_valid_e)
sleep_data_valid_true = filter_and_limit_data(df, 2, 5, db_num_valid_s, db_num_valid_e)
numberH_data_valid_true = filter_and_limit_data(df, 3, 5, db_num_valid_s, db_num_valid_e)
clean_data_valid_true = filter_and_limit_data(df, 4, 5, db_num_valid_s, db_num_valid_e)
service_data_valid_true = filter_and_limit_data(df, 5, 5, db_num_valid_s, db_num_valid_e)
print("Количество строк в списке :", len(price_data_false))
print("Количество строк в списке :", len(sleep_data_false))
print("Количество строк в списке :", len(numberH_data_false))
print("Количество строк в списке :", len(clean_data_false))
print("Количество строк в списке :", len(service_data_false))
print("Количество строк в списке :", len(price_data_valid_false))
print("Количество строк в списке :", len(sleep_data_valid_false))
print("Количество строк в списке :", len(numberH_data_valid_false))
print("Количество строк в списке :", len(clean_data_valid_false))
print("Количество строк в списке :", len(service_data_valid_false))
print("Количество строк в списке :", len(price_data_true))
print("Количество строк в списке :", len(sleep_data_true))
print("Количество строк в списке :", len(numberH_data_true))
print("Количество строк в списке :", len(clean_data_true))
print("Количество строк в списке :", len(service_data_true))
print("Количество строк в списке :", len(price_data_valid_true))
print("Количество строк в списке :", len(sleep_data_valid_true))
print("Количество строк в списке :", len(numberH_data_valid_true))
print("Количество строк в списке :", len(clean_data_valid_true))
print("Количество строк в списке :", len(service_data_valid_true))


def filter_and_limit_data_all(df, column_index, value, start_index, end_index):
    filtered_data = df[df.iloc[:, column_index] == value]  # Фильтрация данных: столбец N, где значение равно N
    filtered_data = filtered_data.iloc[:, 6]  # Сохраняем 6-й столбец из отфильтрованных данных
    filtered_data = filtered_data.tolist()  # Преобразуем столбец в список
    filtered_data = filtered_data[start_index:end_index]  # Ограничиваем в указанных границах
    return filtered_data


data_true = filter_and_limit_data(df, 0, 5, 0, db_num)
data_true_valid = filter_and_limit_data(df, 0, 5, db_num_valid_s, db_num_valid_e)
print("Количество строк в полож:", len(data_true))
print("Количество строк в полож вали:", len(data_true_valid))
data_false = filter_and_limit_data(df, 0, 1, 0, db_num)
data_false_valid = filter_and_limit_data(df, 0, 1, db_num_valid_s, db_num_valid_e)
print("Количество строк в нег:", len(data_false))
print("Количество строк в нег вали:", len(data_false_valid))


def process_data_lem(data):
    print(" текст data ", data)
    data = [remove_digits_and_punctuation(text) for text in data]
    print(" текст после  удаления знаков", data)
    data = [remove_stop_words(text) for text in data]
    print(" текст после  удаления стоп слов", data)
    data = [lemmatize(text) for text in data]
    print(" текст после  леммы", data)

    return data


def process_data_stem(data):
    data = [remove_digits_and_punctuation(text) for text in data]
    data = [remove_stop_words(text) for text in data]
    data = [stem(text) for text in data]
    return data


# data true
count_data_true = len(data_true)
count_data_false = len(data_false)
# цена price
count_price_data_true = len(price_data_true)
count_price_data_false = len(price_data_false)
# качество сна sleep
count_sleep_data_true = len(sleep_data_true)
count_sleep_data_false = len(sleep_data_false)
# номер numberH
count_numberH_data_true = len(numberH_data_true)
count_numberH_data_false = len(numberH_data_false)
# чистота clean
count_clean_data_true = len(clean_data_true)
count_clean_data_false = len(clean_data_false)
# сервис service
count_service_data_true = len(service_data_true)
count_service_data_false = len(service_data_false)
work_data = process_data_lem(data_true + data_false +
                             price_data_true + price_data_false +
                             sleep_data_true + sleep_data_false +
                             numberH_data_true + numberH_data_false +
                             clean_data_true + clean_data_false +
                             service_data_true + service_data_false)
if save==1:
    print("обработанный текст сохранен")
    with open('models\\GRU_LEM\\GRU_LEM_work_data_processed_data.txt', 'w', encoding='utf-8') as file:
        for text in work_data:
            file.write(text + '\n')
else:
    print("обработанный текст не сохранялся")


# данные valid
count_data_true_valid = len(data_true_valid)
count_data_false_valid = len(data_false_valid)
count_price_data_valid_true = len(price_data_valid_true)
count_price_data_valid_false = len(price_data_valid_false)
count_sleep_data_valid_true = len(sleep_data_valid_true)
count_sleep_data_valid_false = len(sleep_data_valid_false)
count_numberH_data_valid_true = len(numberH_data_valid_true)
count_numberH_data_valid_false = len(numberH_data_valid_false)
count_clean_data_valid_true = len(clean_data_valid_true)
count_clean_data_valid_false = len(clean_data_valid_false)
count_service_data_valid_true = len(service_data_valid_true)
count_service_data_valid_false = len(service_data_valid_false)
work_data_valid = process_data_lem(data_true_valid + data_false_valid +
                                   price_data_valid_true + price_data_valid_false +
                                   sleep_data_valid_true + sleep_data_valid_false +
                                   numberH_data_valid_true + numberH_data_valid_false +
                                   clean_data_valid_true + clean_data_valid_false +
                                   service_data_valid_true + service_data_valid_false)

trer= (data_true + data_false + price_data_true + price_data_false +
                             sleep_data_true + sleep_data_false +
                             numberH_data_true + numberH_data_false +
                             clean_data_true + clean_data_false +
                             service_data_true + service_data_false  + data_true_valid + data_false_valid +
                                   price_data_valid_true + price_data_valid_false +
                                   sleep_data_valid_true + sleep_data_valid_false +
                                   numberH_data_valid_true + numberH_data_valid_false +
                                   clean_data_valid_true + clean_data_valid_false +
                                   service_data_valid_true + service_data_valid_false)

word_count = 0
char_count = 0
char_count_no_space = 0
word_count_no_space = 0

for text in trer:
    # Подсчет количества слов в тексте
    word_count += len(text.split())
    # Подсчет количества символов в тексте (включая пробелы)
    char_count += len(text)
    # Подсчет количества символов в тексте (без пробелов)
    char_count_no_space += len(text.replace(" ", ""))

print("Количество слов в тексте:", word_count)
print("Количество символов в тексте (включая пробелы):", char_count)
print("Количество символов в тексте (без пробелов):", char_count_no_space)
if save==1:
    print("обработанный проверочный текст сохранен")
    with open('models\\GRU_lem\\GRU_lem_work_data_valid_processed_data.txt', 'w', encoding='utf-8') as file:
        for text in work_data_valid:
            file.write(text + '\n')
else:
    print("обработанный проверочный текст не сохранялся")

all_work_data = work_data+work_data_valid
tokenizer = Tokenizer(filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(all_work_data)

max_sequence_length = max(len(text.split()) for text in all_work_data)
print(max_sequence_length,"это самая длинная строка а ещё это max_text_len")
maxWordsCount = len(tokenizer.word_index) + 1
dist = list(tokenizer.word_counts.items())
# print(dist[:10])
total_words = sum(tokenizer.word_counts.values())
word_counts_with = [(word, count, count/total_words) for word, count in dist]
df = pd.DataFrame(word_counts_with, columns=['Слово', 'Количество', 'Частота'])
# Записываем DataFrame в файл Excel
excel_file_path1 = 'C:\\Users\\user\\PycharmProjects\\sentimenLSTM\\models\\GRU_LEM\\GRU_LEM_word_count.xlsx'
df.to_excel(excel_file_path1, index=False)

max_text_len = max(len(text.split()) for text in all_work_data)
data = tokenizer.texts_to_sequences(work_data)
data_pad = pad_sequences(data, maxlen=max_text_len)
data2 = tokenizer.texts_to_sequences(work_data_valid)
data_pad2 = pad_sequences(data2, maxlen=max_text_len)

X = data_pad
Y = np.array(
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * count_data_true +          [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * count_data_false +
    [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * count_price_data_true+     [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * count_price_data_false +
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] * count_sleep_data_true +    [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * count_sleep_data_false +
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]] * count_numberH_data_true +  [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * count_numberH_data_false +
    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]] * count_clean_data_true +    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]] * count_clean_data_false +
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] * count_service_data_true +  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * count_service_data_false)
print(X[:2, :], Y[:2, :])
indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]

X2 = data_pad2
Y2 = np.array(
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * count_data_true_valid + [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * count_data_false_valid +
    [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * count_price_data_valid_true+ [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * count_price_data_valid_false +
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] * count_sleep_data_valid_true + [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * count_sleep_data_valid_false +
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]] * count_numberH_data_valid_true + [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * count_numberH_data_valid_false +
    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]] * count_clean_data_valid_true + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]] * count_clean_data_valid_false +
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] * count_service_data_valid_true + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * count_service_data_valid_false)
indeces2 = np.random.choice(X2.shape[0], size=X2.shape[0], replace=False)
X2 = X2[indeces2]
Y2 = Y2[indeces2]


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
print("max_text_len",max_text_len)
print("maxWordsCount",maxWordsCount)

model = Sequential()
model.add(Embedding(maxWordsCount, 300, input_length=max_text_len))
model.add(GRU(256, return_sequences=True))
model.add(BatchNormalization())
model.add(GRU(128, return_sequences=True))
model.add(BatchNormalization())
model.add(GRU(96, return_sequences=True))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(GRU(64, return_sequences=True))
model.add(BatchNormalization())
model.add(GRU(32))
model.add(Dense(12, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m],
              optimizer=Adam(0.0001))
start_time = time.time()  # время обучения
history = model.fit(X, Y, validation_data=(X2, Y2), batch_size=10, epochs=epoch, callbacks=[tensorboard_callback])
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


if save==1:
    print("сохранение модели")
    tokenizer.max_sequence_length = max_sequence_length
    with open('models\\GRU_LEM\\gru_lem_tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.save_weights('models\\GRU_LEM\\STM_LEM_Weights.keras')
    model.save('models\\GRU_LEM\\GRU_LEM_model.keras')
    with open('models\\GRU_LEM\\gru_lem_tokenizer_index.pickle', 'wb') as handle:
        pickle.dump(tokenizer.word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('models\\GRU_LEM\\gru_lem_training_history.pickle', 'wb') as handle:
        pickle.dump(model.history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print("модель не сохранена")

train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_f1_values = history.history['f1_m']
val_f1_values = history.history['val_f1_m']
train_precision_values = history.history['precision_m']
val_precision_values = history.history['val_precision_m']
train_recall_values = history.history['recall_m']
val_recall_values = history.history['val_recall_m']

epochs = range(1, len(train_loss) + 1)
# График функции потерь
plt.figure(figsize=(18, 10))
plt.subplot(3, 3, 1)
plt.plot(epochs, train_loss, 'bo-', label='Потери при обучении')
plt.plot(epochs, val_loss, 'ro-', label='Потери при валидации')
plt.title('Потери при обучении и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
# Графики точности
plt.subplot(3, 3, 2)
plt.plot(epochs, train_accuracy, 'bo-', label='Точность обучения')
plt.plot(epochs, val_accuracy, 'ro-', label='Точность валидации')
plt.title('Точность обучения и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
# Графики f1
plt.subplot(3, 3, 4)
plt.plot(epochs, train_f1_values, 'bo-', label='Обучающая F1-оценка')
plt.plot(epochs, val_f1_values, 'ro-', label='Валидационная F1-оценка')
plt.title('Обучающая и Валидационная F1-оценка')
plt.xlabel('Эпохи')
plt.ylabel('F1-оценка')
plt.legend()
# Графики precision
plt.subplot(3, 3, 5)
plt.plot(epochs, train_precision_values, 'bo-', label='Точность обучения по категории')
plt.plot(epochs, val_precision_values, 'ro-', label='Точность валидации по категории')
plt.title('Точность обучения и валидации по категории')
plt.xlabel('Эпохи')
plt.ylabel('Точность по категории')
plt.legend()
# Графики recall
plt.subplot(3, 3, 6)
plt.plot(epochs, train_recall_values, 'bo-', label='Полнота обучения')
plt.plot(epochs, val_recall_values, 'ro-', label='Полнота на валидации')
plt.title('Полнота обучения и валидации')
plt.xlabel('Эпохи')
plt.ylabel('Полнота')
plt.legend()
# Получаем предсказанные вероятности классов для обучающего набора данных
y_pred_train = model.predict(X)
# Вычисляем кривую ROC и AUC-ROC для первого класса
fpr_0, tpr_0, _ = roc_curve(Y[:, 0], y_pred_train[:, 0])
auc_roc_0 = roc_auc_score(Y[:, 0], y_pred_train[:, 0])
fpr_macro, tpr_macro, _ = roc_curve(Y.ravel(), y_pred_train.ravel())
auc_roc_macro = auc(fpr_macro, tpr_macro)
fpr_micro, tpr_micro, _ = roc_curve(Y.flatten(), y_pred_train.flatten())
auc_roc_micro = auc(fpr_micro, tpr_micro)
# Вычисляем кривую ROC и AUC-ROC для второго класса
fpr_1, tpr_1, _ = roc_curve(Y[:, 1], y_pred_train[:, 1])
auc_roc_1 = roc_auc_score(Y[:, 1], y_pred_train[:, 1])
#print("жопа ",fpr_1)
# Строим график ROC для первого и второго
plt.subplot(3, 3, 7)
plt.plot(fpr_0, tpr_0, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_roc_0)
plt.plot(fpr_1, tpr_1, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % auc_roc_1)
plt.plot(fpr_macro, tpr_macro, color='green', lw=2, linestyle='--', label='Macro-average ROC (AUC = %0.2f)' % auc_roc_macro)
plt.plot(fpr_micro, tpr_micro, color='orange', lw=2, linestyle='--', label='Micro-average ROC (AUC = %0.2f)' % auc_roc_micro)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
fpr_2, tpr_2, _ = roc_curve(Y[:, 2], y_pred_train[:, 2])
auc_roc_2 = roc_auc_score(Y[:, 2], y_pred_train[:, 2])
# Вычисляем кривую ROC и AUC-ROC для второго класса
fpr_3, tpr_3, _ = roc_curve(Y[:, 3], y_pred_train[:, 3])
auc_roc_3 = roc_auc_score(Y[:, 3], y_pred_train[:, 3])
plt.subplot(3, 3, 8)
plt.plot(fpr_2, tpr_2, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_roc_2)
plt.plot(fpr_3, tpr_3, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % auc_roc_3)
plt.plot(fpr_macro, tpr_macro, color='green', lw=2, linestyle='--', label='Macro-average ROC (AUC = %0.2f)' % auc_roc_macro)
plt.plot(fpr_micro, tpr_micro, color='orange', lw=2, linestyle='--', label='Micro-average ROC (AUC = %0.2f)' % auc_roc_micro)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
y_true_classes = np.argmax(Y, axis=1)
y_pred_binary = np.argmax(y_pred_train, axis=1)
conf_matrix = confusion_matrix(y_true_classes, y_pred_binary)
plt.subplot(3, 3, 9)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['общая Н', 'общая П','Цена Н', 'цена П','сон Н', 'сон П','номер Н', 'номер П','чистота Н', 'чистота П','обслуживание Н', 'обслуживание П'],
            yticklabels=['Общая Н', 'Общая П','Цена Н', 'цена П','сон Н', 'сон П','номер Н', 'номер П','чистота Н', 'чистота П','обслуживание Н', 'обслуживание П'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')
plt.tight_layout()
plt.show()

scores = model.evaluate(X, Y, batch_size=10)
print("Точность на тестовых данных: %.2f%%" % (scores[1] * 100))
print("Recall on test data: %.2f%%" % (scores[2] * 100))
print("F1 Score on test data: %.2f%%" % (scores[3] * 100))
print("Precision on test data: %.2f%%" % (scores[4] * 100))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Обучение заняло {elapsed_time} секунд.")

def format(value):
    value = str(value)[:str(value).index('.')]
    return value

if not data_true_valid:
    print("Проверочная дата положительная пустая")
else:
    print("Проверочная дата положительная не пустая")
    jojo = [remove_digits_and_punctuation(text) for text in data_true_valid]
    jojo = [remove_stop_words(text) for text in jojo]
    jojo = [lemmatize(text) for text in jojo]
    data123 = tokenizer.texts_to_sequences(jojo)
    data_pad1 = pad_sequences(data123, maxlen=max_text_len)
    press = model.predict(data_pad1)
    press_sent = press[:, :2]  # сентимент
    press_sent1 = press_sent * 10
    a = press_sent1
    press_price = press[:, 2:4]
    press_price1 = press_price * 10
    ab = press_price1
    press_sleep = press[:, 4:6]
    press_sleep1 = press_sleep * 10
    abc = press_sleep1
    press_numberH = press[:, 6:8]
    press_numberH1 = press_numberH * 10
    abcd = press_numberH1
    press_clean = press[:, 8:10]
    press_clean1 = press_clean * 10
    abcde = press_clean1
    press_service = press[:, 10:12]
    press_service1 = press_service * 10
    abcdef = press_service1
    print(len(jojo))
    rows = []
    for i in range(len(jojo)):
        sentiment_press = "Позитивный" if press_sent[i][0] > press_sent[i][1] else "Негативный"
        press_sent2 = max(a[i])
        sentiment_price = "Позитивный" if press_price[i][0] > press_price[i][1] else "Негативный"
        press_price2 = max(ab[i])
        sentiment_sleep = "Позитивный" if press_sleep[i][0] > press_sleep[i][1] else "Негативный"
        press_sleep2 = max(abc[i])
        sentiment_numberH = "Позитивный" if press_numberH[i][0] > press_numberH[i][1] else "Негативный"
        press_numberH2 = max(abcd[i])
        sentiment_clean = "Позитивный" if press_clean[i][0] > press_clean[i][1] else "Негативный"
        press_clean2 = max(abcde[i])
        sentiment_service = "Позитивный" if press_service[i][0] > press_service[i][1] else "Негативный"
        press_service2 = max(abcdef[i])

        rows.append({
            'вероятности сент': press_sent[i],
            'процент сент': format(press_sent2),
            'Сентимент общий': sentiment_press,

            'вероятности цены': press_price[i],
            'процент цены': format(press_price2),
            'Сентимент цены': sentiment_price,

            'вероятности sleep': press_sleep[i],
            'процент sleep': format(press_sleep2),
            'Сентимент sleep': sentiment_sleep,

            'вероятности numberH': press_numberH[i],
            'процент numberH': format(press_numberH2),
            'Сентимент numberH': sentiment_numberH,

            'вероятности clean': press_clean[i],
            'процент clean': format(press_clean2),
            'Сентимент clean': sentiment_clean,

            'вероятности service': press_service[i],
            'процент service': format(press_service2),
            'Сентимент service': sentiment_service,

            'дата': data_true_valid[i]
        })
    df = pd.DataFrame(rows)
    excel_file_path = 'C:\\Users\\user\\PycharmProjects\\sentimenLSTM\\models\\GRU_LEM\\GRU_LEM_data_true_valid.xlsx'
    df.to_excel(excel_file_path, index=False)


if not data_false_valid:
    print("Проверочная дата негативная пустая")
else:
    print("Проверочная дата негативная не пустая")
    dio = [remove_digits_and_punctuation(text) for text in data_false_valid]
    dio = [remove_stop_words(text) for text in dio]
    dio = [lemmatize(text) for text in dio]
    data1234 = tokenizer.texts_to_sequences(dio)
    data_pad12 = pad_sequences(data1234, maxlen=max_text_len)
    press = model.predict(data_pad12)
    press_sent = press[:, :2]  # сентимент
    press_sent1 = press_sent * 10
    a = press_sent1
    press_price = press[:, 2:4]
    press_price1 = press_price * 10
    ab = press_price1
    press_sleep = press[:, 4:6]
    press_sleep1 = press_sleep * 10
    abc = press_sleep1
    press_numberH = press[:, 6:8]
    press_numberH1 = press_numberH * 10
    abcd = press_numberH1
    press_clean = press[:, 8:10]
    press_clean1 = press_clean * 10
    abcde = press_clean1
    press_service = press[:, 10:12]
    press_service1 = press_service * 10
    abcdef = press_service1
    rows2 = []
    for i in range(len(dio)):
        sentiment_press = "Позитивный" if press_sent[i][0] > press_sent[i][1] else "Негативный"
        press_sent2 = max(a[i])
        sentiment_price = "Позитивный" if press_price[i][0] > press_price[i][1] else "Негативный"
        press_price2 = max(ab[i])
        sentiment_sleep = "Позитивный" if press_sleep[i][0] > press_sleep[i][1] else "Негативный"
        press_sleep2 = max(abc[i])
        sentiment_numberH = "Позитивный" if press_numberH[i][0] > press_numberH[i][1] else "Негативный"
        press_numberH2 = max(abcd[i])
        sentiment_clean = "Позитивный" if press_clean[i][0] > press_clean[i][1] else "Негативный"
        press_clean2 = max(abcde[i])
        sentiment_service = "Позитивный" if press_service[i][0] > press_service[i][1] else "Негативный"
        press_service2 = max(abcdef[i])

        rows2.append({
            'все': press[i],
            'вероятности сент': press_sent[i],
            'процент сент': format(press_sent2),
            'Сентимент общий': sentiment_press,

            'вероятности цены': press_price[i],
            'процент цены': format(press_price2),
            'Сентимент цены': sentiment_price,

            'вероятности sleep': press_sleep[i],
            'процент sleep': format(press_sleep2),
            'Сентимент sleep': sentiment_sleep,

            'вероятности numberH': press_numberH[i],
            'процент numberH': format(press_numberH2),
            'Сентимент numberH': sentiment_numberH,

            'вероятности clean': press_clean[i],
            'процент clean': format(press_clean2),
            'Сентимент clean': sentiment_clean,

            'вероятности service': press_service[i],
            'процент service': format(press_service2),
            'Сентимент service': sentiment_service,

            'дата': data_false_valid[i]
        })
    df = pd.DataFrame(rows2)
    excel_file_path = 'C:\\Users\\user\\PycharmProjects\\sentimenLSTM\\models\\GRU_LEM\\GRU_LEM_data_false_valid.xlsx'
    df.to_excel(excel_file_path, index=False)
