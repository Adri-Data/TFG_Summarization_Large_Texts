# Imports
import concurrent.futures
import gc
import json
import re
import nltk
import torch
import whisperx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from accelerate import Accelerator
from transformers import (AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoTokenizer, MarianMTModel, MarianTokenizer, PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments)
from bert_score import score as bert_score
from datasets import load_dataset
from pytube import YouTube
from rouge_score import rouge_scorer
from summarizer import Summarizer
from nltk.translate import meteor_score
from nltk.translate.bleu_score import sentence_bleu
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def initialize_model_and_tokenizer(model_name):

    # Cargar el tokenizador y el modelo
    if model_name== "t5-base":
        tokenizer = PegasusTokenizer.from_pretrained(model_name)    
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if (model_name=="tuner007/pegasus_summarizer" or model_name=="microsoft/prophetnet-large-uncased"):
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
      model = AutoModelWithLMHead.from_pretrained(model_name)

    # Mover el modelo a la GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, tokenizer, device

def sumary_final(inputs):
  summary_ids = model.generate(inputs, max_length=200, min_length=100, length_penalty=0.1, num_beams=16)
  return summary_ids

def transcribe_and_translate(url, source_lang='es', target_lang='en', device='cuda'):
    # Crear un objeto YouTube
    yt = YouTube(url)
    # Obtener el stream de audio de mayor calidad
    audio_stream = yt.streams.get_audio_only()
    # Descargar el audio
    audio_file = audio_stream.download()
    # Transcribir el audio usando whisperx
    model = whisperx.load_model("large-v2", device, compute_type="float32")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)
    # Alinear el texto con el audio
    model_a, metadata = whisperx.load_align_model(language_code=source_lang, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # Traducir el texto al inglés usando MarianMT
    modelo_nombre = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    modelo = MarianMTModel.from_pretrained(modelo_nombre).to(device)
    tokenizador = MarianTokenizer.from_pretrained(modelo_nombre)

    def traducir_segmento(segmento):
        id, texto = segmento
        # Codificar el texto y generar la traducción
        texto_tokenizado = tokenizador.prepare_seq2seq_batch([texto], return_tensors='pt').to(device)
        traduccion_ids = modelo.generate(**texto_tokenizado)
        traduccion = tokenizador.decode(traduccion_ids[0], skip_special_tokens=True)
        return id, traduccion

    # Codificar el texto y generar la traducción
    with ThreadPoolExecutor() as executor:
        segmentos = [(i, diccionario['text']) for i, diccionario in enumerate(result["segments"])]
        traducciones = list(executor.map(traducir_segmento, segmentos))
        # Ordenar las traducciones por id y juntarlas
        texto_traducido = " ".join(traduccion for id, traduccion in sorted(traducciones))
        # Devolver el texto traducido
    return texto_traducido


def divide_texto(texto, tokenizer, longitud_max=512):

    nltk.download('punkt', quiet=True)

    # Divide el texto en oraciones utilizando NLTK
    oraciones = nltk.tokenize.sent_tokenize(texto)

    trozos = []
    trozo_actual = ''
    for i, parrafo in enumerate(oraciones):
        if len(tokenizer.encode(trozo_actual + '\n' + parrafo)) <= longitud_max:
            trozo_actual += '\n' + parrafo
        else:
            trozos.append((i, trozo_actual))
            trozo_actual = parrafo
    trozos.append((i, trozo_actual))
    return trozos

def resumir_texto(trozo, model, tokenizer, device):
    id, texto = trozo
    # Codificar el texto y generar el resumen
    inputs = tokenizer.encode("summarize: " + texto, return_tensors='pt', max_length=512, truncation=True, padding='longest').to(device)
    summary_ids = model.generate(inputs, max_length=100, min_length=20, length_penalty=3, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return id, resumen

def resumir_texto_final(trozo, model, tokenizer, device):
    id, texto = trozo
    # Codificar el texto y generar el resumen
    inputs = tokenizer.encode("summarize: " + texto, return_tensors='pt', truncation=True, padding='longest').to(device)
    summary_ids = sumary_final(inputs)
    resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return id, resumen


def resumir_texto_paralelo(texto, model, tokenizer, device, max_length, print_option="no"):
    # Dividir el texto en trozos
    if len(tokenizer.encode(texto)) <= max_length:
        return resumir_texto_final((None, texto), model, tokenizer, device)[1]

    trozos = divide_texto(texto, tokenizer, max_length)

    # Si el número de tokens es menor que max_length, llama a resumir_texto_final


    resumenes = {}

    # Crea un pool de trabajadores para ejecutar resumir_texto en paraleloº
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Mapea resumir_texto a cada trozo de texto
        future_to_trozo = {executor.submit(resumir_texto, trozo, model, tokenizer, device): trozo for trozo in trozos}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_trozo)):
            trozo = future_to_trozo[future]
            try:
                id, resumen = future.result()
            except Exception as exc:
                print(f'El trozo {trozo} generó una excepción: {exc}')
            else:
                resumenes[id] = resumen
                porcentaje_completado = ((i + 1) / len(trozos)) * 100
                if (print_option!="no"):
                  print(f'Proceso completado: {i + 1}. Porcentaje completado: {porcentaje_completado}%')
                  print(f'Resumen: {resumen}')

    # Ordenar los resúmenes por id y juntarlos
    texto_resumen = ' '.join(resumen for id, resumen in sorted(resumenes.items()))

    # Si los tokens de texto_resumen son más que max_length, llama a resumir_texto_paralelo de forma recursiva
    texto_resumen = resumir_texto_paralelo(texto_resumen, model, tokenizer, device, max_length, print_option)

    return texto_resumen

def generar_resumen_extractivo(texto, ratio, algoritmo="pagerank"):
    # Inicializa el modelo
    model = Summarizer()

    # Genera el resumen
    resumen = model(texto, ratio=ratio, algorithm=algoritmo)

    return resumen

# URL del video de YouTube para el test
url = "https://www.youtube.com/watch?v=vgbMR0lDQBI"  # Reemplaza esto con la URL de tu video

# Longitud máxima del resumen para el test
max_length = 400  # Reemplaza esto con la longitud máxima que desees

# Transcribir y traducir el video al inglés
translated_text = transcribe_and_translate(url)

from concurrent.futures import ThreadPoolExecutor

model_names = ["tuner007/pegasus_summarizer", "facebook/bart-large-cnn", "t5-base", "microsoft/prophetnet-large-uncased"]
text = translated_text
reduced_text = generar_resumen_extractivo(text, ratio=0.3)

# Para cada modelo
for model_name in model_names:
    # Carga el tokenizador y el modelo
    model, tokenizer, device = initialize_model_and_tokenizer(model_name)

    #resumen 1 model
    _, summary_original = resumir_texto_final([_, translated_text], model, tokenizer, device)

    #resume pipeline
    summary_pipeline = resumir_texto_paralelo(text, model, tokenizer, device, max_length=400, print_option="no")

    #resumen 1 model extractive
    _, summary_original_extracted = resumir_texto_final([_, reduced_text], model, tokenizer, device)

    #resume pipeline
    summary_pipeline_extracted = resumir_texto_paralelo(reduced_text, model, tokenizer, device, max_length=400, print_option="no")

    print(f"Model: {model_name}")

    print("_________________________________________________________________\n\n")

    print(f"\n Generated Summary without the pipeline: {summary_original}")

    print(f"\n Generated Summary with the pipeline: {summary_pipeline}")

    print(f"\n Generated Summary with extractive summarization: {summary_original_extracted}")

    print(f"\n Generated Summary with pipeline and extractive summarization: {summary_pipeline_extracted}")

    print("_________________________________________________________________\n\n")
    del model
    del tokenizer
    torch.cuda.empty_cache()