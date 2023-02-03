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

from config import TOKEN  #получаем токен из соседнего файла в папке при запуске в IDE


logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)

# For example use simple MemoryStorage for Dispatcher.
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

correct_words = open("dictionary.txt", "r").readlines()

class UserData(StatesGroup):
    """Для хранения состояний.
    Наследуется от StatesGroup. Использует State."""
    individual_list = State()

# Поиск областей с текстом на изображениях с помощью Paddle
# Функция для поиска координат слов и отрисовке боксов на картинке
def detect_img(paths:list, ocr:str):

  reader = None
  results = []

  if ocr == 'easyocr':
    pass # reader = easyocr.Reader(['ru'], detect_network = 'dbnet18')
  elif ocr == 'paddleocr':
    reader = PaddleOCR(use_angle_cls=True, lang='ru')

  if reader is None:
    raise Exception('Вы не указали верную модель')
  
  for path in paths:

    image = cv2.imread(path)
    photo_name = path.split('/')[-1]

    if ocr == 'easyocr':
      result = reader.readtext(path)
    else:
      temp_result = reader.ocr(path, cls=True)

      result = []
      for i in temp_result[0]:
        i = (i[0], i[1][0], i[1][1])
        result.append(i)

    for (bbox, text, prob) in result:

      (tl, tr, br, bl) = bbox
      tl = (int(tl[0]), int(tl[1]))
      tr = (int(tr[0]), int(tr[1]))
      br = (int(br[0]), int(br[1]))
      bl = (int(bl[0]), int(bl[1]))

      cv2.rectangle(image, tl, br, (0, 255, 0), 2)

    os.mkdir(f'{ocr}_results/image_detection/{photo_name}')
    cv2.imwrite(f'{ocr}_results/image_detection/{photo_name}/{photo_name}', image)
    results.append(result)

  return results


  # Вырезать слова из фото
def get_detected_words_easyocr(paths:list, ocr:str):

  results = detect_img(paths, ocr)

  for index in range(len(paths)):

    image = cv2.imread(paths[index])
    result = results[index]
    photo_name = paths[index].split('/')[-1]

    os.mkdir(f'{ocr}_results/detected_words/{photo_name}')
    for index, (bbox, text, prob) in enumerate(result):
      word = image[int((bbox[0][1])):int((bbox[2][1])), int((bbox[0][0])):int((bbox[2][0])), :]
      if (word.shape[0] != 0) and (word.shape[1] != 0) and (word.shape[2] != 0):
        cv2.imwrite(f'{ocr}_results/detected_words/{photo_name}/{index}_{photo_name}', word)
  return photo_name


  # Получить пути к полученным словам
def get_img(ocr:str, photo_name):
  path = f'{ocr}_results/detected_words/{photo_name}'
  # globals()['list_%s' % ocr] = sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))
  # globals()['path_%s' % ocr] = path
  list_paddleocr = sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))
  path_paddleocr = path
  return list_paddleocr, path_paddleocr


  # Запускает предыдущие функции
def main(paths:list):
  # try:
  #   shutil.rmtree('easyocr_results')
  #   os.mkdir('easyocr_results')
  #   os.mkdir('easyocr_results/image_detection')
  #   os.mkdir('easyocr_results/detected_words')   
  # except:
  #   os.mkdir('easyocr_results')
  #   os.mkdir('easyocr_results/image_detection')
  #   os.mkdir('easyocr_results/detected_words')

  try:
    shutil.rmtree('paddleocr_results')
    os.mkdir('paddleocr_results')
    os.mkdir('paddleocr_results/image_detection')
    os.mkdir('paddleocr_results/detected_words')
  except:
    os.mkdir('paddleocr_results')
    os.mkdir('paddleocr_results/image_detection')
    os.mkdir('paddleocr_results/detected_words')

  # photo_name = get_detected_words_easyocr(paths, 'easyocr')
  photo_name = get_detected_words_easyocr(paths, 'paddleocr')
  
  # get_img('easyocr', photo_name)
  list_paddleocr, path_paddleocr = get_img('paddleocr', photo_name)

  return list_paddleocr, path_paddleocr


# main(['/home/rushana/Final_project/lecho_C.jpg'])

def text_recognition(list_paddleocr, path_paddleocr):
  
  ocr = PaddleOCR(rec_model_dir='./', rec_char_dict_path='rus_chars.txt')

  result_easyocr = []
  result_paddleocr = []

  # for im in list_easyocr:
  #     img_path = f"{path_easyocr}/{im}"
  #     result = ocr.ocr(img_path, det=False, cls=False)
  #     # print(result)
  #     for idx in range(len(result)):
  #       res = result[idx]
  #       result_easyocr.append(res[0][0].lower())


  for im in list_paddleocr:
      img_path = f"{path_paddleocr}/{im}"
      result = ocr.ocr(img_path, det=False, cls=False)
      for idx in range(len(result)):
        res = result[idx]
        result_paddleocr.append(res[0][0].lower())

  # svtr_result = list(set(result_easyocr + result_paddleocr))
  svtr_result = result_paddleocr

  return svtr_result



# Исправление распознанных слов с помощью FuzzyWuzzy
# функция для определения является ли распознанное слово числом
def is_number(str):
    try:
      float(str)
      return True
    except:
      return False



def run_fuzzywuzzy(found_words: list):
  # correct_words = open("dictionary.txt", "r").readlines()
  final_list = []
  for word in found_words:
    if is_number(word) == False:
      high_score = 50
      best_word = ''
      for cor_word in correct_words:
        result = fuzz.token_sort_ratio(word, cor_word[:-1])
        if result > high_score:
          high_score = result
          best_word = cor_word[:-1]
        pass
      if best_word != '':
        if best_word not in final_list:
          final_list.append(best_word)
  return final_list



#Распознавание текста на изображении
def image_to_text(paths:list):
  list_paddleocr, path_paddleocr = main(paths)
  found_words = text_recognition(list_paddleocr, path_paddleocr)
  final_list = run_fuzzywuzzy(found_words)
  # print(final_list)
  return final_list

@dp.message_handler(commands=['start'])          # декоратор (функция, принимающая в виде параметра другую функцию), регистрирует обработчик комманды 'start'
async def send_welcome(message:types.Message, state: FSMContext):   # асинхронная функция, получающая на вход сообщение. Код выполняется дальше, если функция ждет ввода.
    user_name = message.from_user.full_name      # инициализируется переменная внутри функции, ей присваивается значение из профиля юзера.
    user_id = message.from_user.id
    text_start = f"Здравствуйте, {user_name}! Сфотографируйте состав на упаковке продукта или загрузите сюда фотографию состава. Если хотите добавить нежелательные ингредиенты, напишите /add" # формируется текст приветствия
    async with state.proxy() as data:
        data['list_out'] = ['мука', 'пальмовое', 'арахис', 'е102', 'е103', 'е104', 'е105', 'е107', 'е110', 'е111', 'е120', 'е121', 'е122', 'е123', 'е124', 'е125', 'е126', 'е127', 'е128', 'е129', 'е130', 'е131', 'е132', 'е133', 'е141', 'е142', 'е150', 'е151', 'е152', 'е153', 'е154', 'е155', 'е160', 'е166', 'е171', 'е173', 'е174', 'е175', 'е180', 'е181', 'е182', 'е200', 'е201', 'е209', 'е210', 'е211', 'е212', 'е213', 'е214', 'е215', 'е216', 'е217', 'е218', 'е219', 'е220', 'е221', 'е222', 'е223', 'е224', 'е225', 'е226', 'е227', 'е228', 'е230', 'е231', 'е232', 'е233', 'е234', 'е235', 'е236', 'е237', 'е238', 'е239', 'е240', 'е241', 'е242', 'е249', 'е250', 'е251', 'е252', 'е261', 'е262', 'е263', 'е264', 'е270', 'е280', 'е281', 'е282', 'е283', 'е284', 'е285', 'е296', 'е297', 'е310', 'е311', 'е312', 'е320', 'е321', 'е330', 'е338', 'е339', 'е340', 'е341', 'е343', 'е400', 'е401', 'е402', 'е403', 'е404', 'е405', 'е450', 'е451', 'е452', 'е453', 'е454', 'е461', 'е462', 'е463', 'е465', 'е466', 'е477', 'е501', 'е502', 'е503', 'е510', 'е513', 'е527', 'е620', 'е621', 'е622', 'е625', 'е626', 'е627', 'е628', 'е629', 'е630', 'е631', 'е632', 'е633', 'е634', 'е635', 'е636', 'е637', 'е900', 'е901', 'е902', 'е903', 'е904', 'е905', 'е906', 'е907', 'е908', 'е909', 'е910', 'е911', 'е912', 'е913', 'е914', 'е916', 'е917', 'е918', 'е919', 'е922', 'е923', 'е924', 'е925', 'е926', 'е927', 'е928', 'е929', 'е930', 'е938', 'е939', 'е941', 'е942', 'е943', 'е944', 'е945', 'е946', 'е948', 'е950', 'е951', 'е952', 'е953', 'е954', 'е957', 'е958', 'е959', 'е965', 'е966', 'е967', 'е999']                               # отпрвляем в словарь, работающий между функциями, список вредных продуктов
    logging.info(f"{user_name=} {user_id=} sent message: {message.text}")  #передача в консоль переменные имени, user_id, отправляемое сообщение
    await message.answer(text_start)                    # аналог return, но функция после выдачи информации может продолжаться дальше, если дальше есть её код.
    await state.update_data(individual_list='empty')

@dp.message_handler(commands=['help'], state="*")
async def process_help_command(message: types.Message):
    await message.reply("Определяю то что есть в составе продуктов. Отправьте фото и следующим сообщением отправьте слова, которые мне нужно найти через пробел.")

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
    t = f"Добавлены в личный список следующие нежелательные ингредиенты: {', '.join(ind_list)}. Загрузите изображение состава продукта или сделайте фото."
    async with state.proxy() as data:              #словарь передается между функциями                      
        ind_list.extend(data['list_out'])
        data['list_out'] = ind_list
    await message.answer(t)


@dp.message_handler(content_types=['text'], state="*")
async def add_message(message:types.Message, state: FSMContext):
    # user_name = message.from_user.full_name
    # user_id = message.from_user.id
    text_string = message.text
    low_string = text_string.lower()
    clean_string = re.sub(r'[^\w\s]','', low_string)
    text_listing = clean_string.split()                   #пользовательский список, полученный в сообщении
    # list2 = list1
    # # async with state.proxy() as data:
    #     #     data['list_out'] = list2
    # async with state.proxy() as data:              #словарь передается между функциями                      
    #     list1.extend(data['list_out'])
    # text = f"Из введенных выше найдены следующие нежелательные ингредиенты: {', '.join(list2)}."
    # await message.answer( text)
                                                  # Декоратор, регистрирующий обработчик события ввода любого сообщения (кроме того, что выше).
# @dp.message_handler(state="*")                             # Если в скобках не указан тип сообщения, от обрабатываются только текстовые сообщения.
# async def send_message(message:types.Message, state: FSMContext): # передаем FSMContext в функцию для сохранения переменных в словаре вне этой фукнции
#     user_name = message.from_user.full_name
#     user_id = message.from_user.id
#     text = message.text
#     list1 = text.lower().split()                   #пользовательский список, полученный в сообщении
    list2 = []
    async with state.proxy() as data:              #словарь передается между функциями
        l = data['list_out']                       #присваеваем значение м
    for i in l:
        if i in text_listing:                             #в list2 получаем пересечение списков вредных и введенных продуктов и записываем его в словарь data['list_out']
            list2.append(i)
    # async with state.proxy() as data:
    #     data['list_out'] = list2
    exit_text = f"Нежелательные ингредиенты: {', '.join(list2)}."
    # logging.info(f"{user_name=} {user_id=} sent message: {message.text}")        #выдается пересечение списков
    await message.answer(exit_text)

@dp.message_handler(content_types=["photo"], state="*")
async def download_photo(message: types.Message, state: FSMContext):
    # list2 = ['сахар', 'мука']
    # async with state.proxy() as data:
    #     list2 = data['list_out']
    await message.photo[-1].download(destination="./")
    # message.photo[-1].download(destination="./")
    path = r'/home/alex/projects/nutritional_assistant/photos/'
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    files = [file for file in files if os.path.isfile(file)]
    path_f = max(files, key=os.path.getctime)
    list1 = image_to_text([path_f])
    list2 = []
    async with state.proxy() as data:              #словарь передается между функциями
        l = data['list_out']                       #присваеваем значение м
    for i in l:
        if i in list1:                             #в list2 получаем пересечение списков вредных и введенных продуктов и записываем его в словарь data['list_out']
            list2.append(i)
    if len(list2) > 0:
        text = f"Нежелательные ингредиенты: {','.join(list2)}."
    else:
        text = f"Нежелательных ингредиентов не обнаружено."
    await message.answer(text)


if __name__ == '__main__':
    executor.start_polling(dp)   #(poll - опрашивать) запуск бота в режиме опроса телеграмм сервера о новых сообщениях.