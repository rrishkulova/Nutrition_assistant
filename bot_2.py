import cv2
import numpy as np
import easyocr
from torch.utils.data import Dataset
import os
import torch
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from paddleocr import PaddleOCR
import shutil

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
  try:
    shutil.rmtree('easyocr_results')
    os.mkdir('easyocr_results')
    os.mkdir('easyocr_results/image_detection')
    os.mkdir('easyocr_results/detected_words')   
  except:
    os.mkdir('easyocr_results')
    os.mkdir('easyocr_results/image_detection')
    os.mkdir('easyocr_results/detected_words')

  try:
    shutil.rmtree('paddleocr_results')
    os.mkdir('paddleocr_results')
    os.mkdir('paddleocr_results/image_detection')
    os.mkdir('paddleocr_results/detected_words')
  except:
    os.mkdir('paddleocr_results')
    os.mkdir('paddleocr_results/image_detection')
    os.mkdir('paddleocr_results/detected_words')

  photo_name = get_detected_words_easyocr(paths, 'easyocr')
  photo_name = get_detected_words_easyocr(paths, 'paddleocr')
  
  get_img('easyocr', photo_name)
  get_img('paddleocr', photo_name)

  return list_easyocr, path_easyocr, list_paddleocr, path_paddleocr

# Распознавание текста на выделенных областях
def text_recognition(list_easyocr, path_easyocr, list_paddleocr, path_paddleocr):
  
  ocr = PaddleOCR(rec_model_dir='/home/rushana/Final_project/CVTR_Tiny_inference', rec_char_dict_path='/home/rushana/Final_project/rus_chars.txt')

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
# функция для определения является ли распознанное слово числом
def is_number(str):
    try:
      float(str)
      return True
    except:
      return False

def run_fuzzywuzzy(found_words: list):
  correct_words = open("correct_words.txt", "r").readlines()
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
  main(paths)
  found_words = text_recognition(list_easyocr, path_easyocr, list_paddleocr, path_paddleocr)
  final_list = run_fuzzywuzzy(found_words)
  print(final_list)

# image_to_text(['/home/rushana/Final_project/lecho_C.jpg'])

image_to_text(['111.jpg'])