import cv2
import numpy as np
import easyocr
from torch.utils.data import Dataset
import os
import torch
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from paddleocr import PaddleOCR, draw_ocr

# Поиск областей с текстом на изображениях с помощью EasyOCR и Paddle
# Функция для поиска координат слов и отрисовке боксов на картинке
def detect_img(paths:list, ocr:str):

  reader = None
  results = []

  if ocr == 'easyocr':
    reader = easyocr.Reader(['ru'], detect_network = 'dbnet18')
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
  globals()['list_%s' % ocr] = sorted(os.listdir(path), key=lambda x: int(x.split('_')[0]))
  globals()['path_%s' % ocr] = path


  # Запускает предыдущие функции
def main(paths:list):
  # os.removedirs('easyocr_results')
  # os.removedirs('paddleocr_results')

  # os.mkdir('easyocr_results')
  # os.mkdir('easyocr_results/image_detection')
  # os.mkdir('easyocr_results/detected_words')

  # os.mkdir('paddleocr_results')
  # os.mkdir('paddleocr_results/image_detection')
  # os.mkdir('paddleocr_results/detected_words')

  photo_name = get_detected_words_easyocr(paths, 'easyocr')
  photo_name = get_detected_words_easyocr(paths, 'paddleocr')
  
  get_img('easyocr', photo_name)
  get_img('paddleocr', photo_name)

  return list_easyocr, path_easyocr, list_paddleocr, path_paddleocr

# main(['/home/rushana/Финальный_проект/lecho_C.jpg'])

# Распознавание текста на выделенных областях
def text_recognition(list_easyocr, path_easyocr, list_paddleocr, path_paddleocr):
  
  ocr = PaddleOCR(rec_model_dir='./', rec_char_dict_path='./rus_chars.txt')

  result_easyocr = []
  result_paddleocr = []

  for im in list_easyocr:
      img_path = f"{path_easyocr}/{im}"
      result = ocr.ocr(img_path, det=False, cls=False)
      # print(result)
      for idx in range(len(result)):
        res = result[idx]
        result_easyocr.append(res[0][0].lower())


  for im in list_paddleocr:
      img_path = f"{path_paddleocr}/{im}"
      result = ocr.ocr(img_path, det=False, cls=False)
      for idx in range(len(result)):
        res = result[idx]
        result_paddleocr.append(res[0][0].lower())

  svtr_result = list(set(result_easyocr + result_paddleocr))

  return svtr_result

# Исправление распознанных слов с помощью FuzzyWuzzy
def fuzzywuzzy(found_words: list):
  final_list = []
  correct_words = ['глюкоза', 'декстроза', 'е102', 'е103', 'е105', 'е110', 'е1105', 'е121', 'е123', 'е125', 'е126', 'е128', 'е130', 'е131', 
                 'е142', 'е151', 'е152', 'е153', 'е154', 'е160', 'е210', 'е211', 'е212', 'е213', 'е214', 'е215', 'е216', 'е216', 'е217', 
                 'е217', 'е219', 'е220', 'е221', 'е222', 'е223', 'е224', 'е225', 'е226', 'е230', 'е230', 'е230', 'е231', 'е231', 'е232', 
                 'е232', 'е233', 'е239', 'е239', 'е240', 'е249', 'е252', 'е280', 'е281', 'е282', 'е283', 'е311', 'е312', 'е313', 'е320', 
                 'е320', 'е321', 'е321', 'е322', 'е330', 'е338', 'е339', 'е340', 'е341', 'е343', 'е405', 'е407', 'е447', 'е450', 'е451', 
                 'е452', 'е453', 'е454', 'е461', 'е462', 'е463', 'е464', 'е465', 'е466', 'е626', 'е627', 'е628', 'е629', 'е630', 'е631', 
                 'е632', 'е633', 'е634', 'е635', 'е951', 'е954', 'ксилит', 'кукурузный сироп', 'мальтоза', 'меласса', 'нектар агавы', 
                 'пальмовое масло', 'патока', 'сахароза', 'сироп топинамбура', 'сорбит', 'сукралоза', 'тростниковый сок', 'фруктоза', 
                 'сахар', 'состав', 'пищевая', "ценность", "молоко", "наполнитель", "смородина", "ягода", "сок", "концентрированный", 
                 "закваска", "углеводы", "жиры", "белки", "срок", "годности", "вода", "питьевая", "сироп", "топинамбура"]
  for word in found_words:
    high_score = 50
    best_word = ''
    for cor_word in correct_words:
      result = fuzz.token_sort_ratio(word, cor_word)
      # print(f'{word} - {cor_word}: {result}')
      if result > high_score:
        high_score = result
        best_word = cor_word
      pass
    if best_word != '':
      if best_word not in final_list:
        final_list.append(best_word)
  return final_list

#Распознавание текста на изображении
def image_to_text(paths:list):
  main(paths)
  found_words = text_recognition(list_easyocr, path_easyocr, list_paddleocr, path_paddleocr)
  final_list = fuzzywuzzy(found_words)
  print(final_list)

image_to_text('./111.jpg')