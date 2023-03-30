import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

import cv2
from torch.utils.data import Dataset
import os
from fuzzywuzzy import fuzz
from paddleocr import PaddleOCR
import shutil
import numpy as np
import json
import re
from nltk.stem.snowball import RussianStemmer

# from config import TOKEN  #получаем токен из соседнего файла в папке при запуске в IDE

TOKEN = '6185293322:AAGi4D_z7ztkscJj0eMzO2t6VyNDTQcXbFQ'
logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

#генерируем словарь для FuzzyWuzzy
correct_words = open("dictionary.txt", "r").readlines()


#генерируем словарь adds_info с нежелательными добавками
with open('adds_info.txt') as file:
  lines = file.read().splitlines() 
adds_info = {}
for line in lines: 
  key, value = line.split(': ')
  adds_info.update({key:value})	


class UserData(StatesGroup):
    """Для хранения состояний.
    Наследуется от StatesGroup. Использует State."""
    individual_list = State()

#прописываем функцию для распознавания слов с помощью библиотеки PaddleOCR
def find_words(img_path:str):
  found_words = []
  ocr = PaddleOCR(rec_model_dir="PaddleOCR_rec", lang='ru', use_angle_cls=True, rec_char_dict_path="rus_chars.txt")
  result = ocr.ocr(img_path, cls=True)
  for idx in range(len(result)):
      res = result[idx]   
      for line in res:
          found_words.append(re.sub("[^А-Яа-я0-9]",'', line[1][0].lower()))
  return found_words


#исходим из предположения, что если распознанное слово содержит в себе цифры, значит это или Е-шка 
#или срок годности/телефон/БЖУ, т.е. преобразовывать его с помощью FuzzyWuzzy не требуется, поэтому создадим проверочную функцию
def is_number(string):
    if re.search('\d+', string) is None:
      return False
    else:
      return True


#функция для "причесывания" слов с помощью FuzzyWuzzy
def run_fuzzywuzzy(found_words: list):
  with_numbers = []
  just_words = []
  for i, word in enumerate(found_words):
    if is_number(word) == True:
      with_numbers.append(word)

    else:
      high_score = 50
      best_word = ''
      for cor_word in correct_words:
        result = fuzz.token_sort_ratio(word, cor_word[:-1])
        if result > high_score:
          high_score = result
          best_word = cor_word[:-1]
        pass
      if best_word != '':
        if best_word not in just_words:
          # with_numbers.append(best_word)
          just_words.append(best_word)
  return with_numbers, just_words   


#финальная функция "изображение -> распознанные слова"
def image_to_text(path:str):
  found_words = find_words(path)
  with_numbers, just_words = run_fuzzywuzzy(found_words)
  return with_numbers, just_words


@dp.message_handler(commands=['start'])          # декоратор (функция, принимающая в виде параметра другую функцию), регистрирует обработчик комманды 'start'
async def send_welcome(message:types.Message, state: FSMContext):   # асинхронная функция, получающая на вход сообщение. Код выполняется дальше, если функция ждет ввода.
    user_name = message.from_user.full_name      # инициализируется переменная внутри функции, ей присваивается значение из профиля юзера.
    user_id = message.from_user.id
    text_start = f"Здравствуйте, {user_name}! Сфотографируйте состав на упаковке продукта или загрузите сюда фотографию состава. Если хотите добавить нежелательные ингредиенты, напишите /add" # формируется текст приветствия
    async with state.proxy() as data:
        data['list_out'] = adds_info # отпрвляем в словарь, работающий между функциями, список вредных продуктов
    logging.info(f"{user_name=} {user_id=} sent message: {message.text}")  #передача в консоль переменные имени, user_id, отправляемое сообщение
    await message.answer(text_start)                    # аналог return, но функция после выдачи информации может продолжаться дальше, если дальше есть её код.
    await state.update_data(individual_list='empty')

@dp.message_handler(commands=['help'], state="*")
async def process_help_command(message: types.Message):
    await message.reply("Определяю то, что есть в составе продукта. Отправьте фото и следующим сообщением отправьте слова, которые мне нужно найти через пробел.")

@dp.message_handler(commands=['add'], state="*")
async def send_message(message:types.Message, state: FSMContext): # передаем FSMContext в функцию для сохранения переменных в словаре вне этой фукнции
    await message.answer("Введите список нежелательных ингредиентов.")
    await UserData.individual_list.set()

@dp.message_handler(content_types=['text'], state=UserData.individual_list)
async def add_message(message:types.Message, state: FSMContext):
    add_text = message.text
    add_text_low = add_text.lower()
    add_text_clean = re.sub(r'[^\w\s]','', add_text_low)
    ind_list = add_text_clean.split()                   #пользовательский список, полученный в сообщении
    ind_dict = {key: key+f' - входит в Ваш личный стоп-лист' for key in ind_list}

    t = f"Добавлены в личный стоп-лист следующие нежелательные ингредиенты: {', '.join(ind_list)}. Загрузите изображение состава продукта или сделайте фото."
    async with state.proxy() as data:              #словарь передается между функциями                           
        ind_dict.update(data['list_out'])
        data['list_out'] = ind_dict
    await message.answer(t)


@dp.message_handler(content_types=['text'], state="*")
async def add_message(message:types.Message, state: FSMContext):
    text_string = message.text
    low_string = text_string.lower()
    clean_string = re.sub(r'[^\w\s]','', low_string)
    text_listing = clean_string.split()                   #пользовательский список, полученный в сообщении
    list2 = []
    async with state.proxy() as data:              #словарь передается между функциями
        l = data['list_out']                       #присваеваем значение м
    for i in l:
        if i in text_listing:                             #в list2 получаем пересечение списков вредных и введенных продуктов и записываем его в словарь data['list_out']
            list2.append(i)
    exit_text = f"Нежелательные ингредиенты: {'; '.join(list2)}."
    await message.answer(exit_text)

@dp.message_handler(content_types=["photo"], state="*")
async def download_photo(message: types.Message, state: FSMContext):
    await message.photo[-1].download(destination="./")
    path = r'/home/rushana/Final_project/9_bots/bot_v1/photos'
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    files = [file for file in files if os.path.isfile(file)]
    path_f = max(files, key=os.path.getctime)
    with_numbers, just_words = image_to_text(path_f)
    list2 = []
    async with state.proxy() as data:              #словарь передается между функциями
        l = data['list_out']                       #присваеваем значение м
    
    #этот блок кода для проверки на наличие в составе ингредиентов, производных от указанных в личном стоп-листе ("арахис" >> "арахиса, арахисовый...")
    stemmer  = RussianStemmer()
    stem_just_words = [stemmer.stem(x) for x in just_words]
    for word in stem_just_words:
      high_score = 60
      best_word = ''
      for cor_word in l:
        result = fuzz.ratio(word, cor_word[:-1])
        if result > high_score:
          high_score = result
          best_word = cor_word[:-1]
          if l[cor_word] not in list2:
            list2.append(l[cor_word])

    for i in with_numbers:
      if i in l and i not in list2:                             #в list2 получаем пересечение списков вредных и введенных продуктов и записываем его в словарь data['list_out']
        list2.append(l[i])

    if len(list2) > 0:
        text = f"Нежелательные ингредиенты: {'; '.join(list2)}."
    else:
        text = f"Нежелательных ингредиентов не обнаружено."
    await message.answer(text)


if __name__ == '__main__':
    executor.start_polling(dp)   #(poll - опрашивать) запуск бота в режиме опроса телеграмм сервера о новых сообщениях.