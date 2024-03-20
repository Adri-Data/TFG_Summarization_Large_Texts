# Imports
import concurrent.futures
import gc
import json
import re
import nltk
import torch
import whisperx
import pandas as pd
import numpy as np
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
  summary_ids = model.generate(inputs, max_length=200, min_length=100, length_penalty=0.1, num_beams=16, no_repeat_ngram_size=3)
  return summary_ids

def transcribe_and_translate(url, source_lang='es', target_lang='en', device='cuda'):
    # Crear un objeto YouTube
    yt = YouTube(url)
    # Obtener el stream de audio de mayor calidad
    audio_stream = yt.streams.get_audio_only()
    # Descargar el audio
    audio_file = audio_stream.download()
    # Transcribir el audio usando whisperx
    model = whisperx.load_model("large-v2", device, compute_type="float16")
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

torch.cuda.empty_cache()
# Inicializa las puntuaciones
scores = {
    "model_name": [],
    "methods": [],
    "ROUGE1-Precision": [],
    "ROUGE1-Recall": [],
    "ROUGE1-F1": [],
    "ROUGE2-Precision": [],
    "ROUGE2-Recall": [],
    "ROUGE2-F1": [],
    "ROUGEL-Precision": [],
    "ROUGEL-Recall": [],
    "ROUGEL-F1": [],
    "BLEU": [],
    "METEOR": [],
    "Cosine Similarity": [],
    "BERTScore": [],
}

# Define los modelos que quieres probar
model_names = ["google-t5/t5-base","tuner007/pegasus_summarizer", "facebook/bart-large-cnn", "microsoft/prophetnet-large-uncased"]
# Define los métodos de resumen
summary_methods = ["original", "pipeline", "original_extracted", "pipeline_extracted"]

# Carga el conjunto de datos booksum
dataset = load_dataset('kmfoda/booksum')

# Inicializa el evaluador de ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
print(model_names)
# Para cada modelo
print("comenzamos a tomar los modelos")
for model_name in model_names:
    print(model_name)
    # Carga el tokenizador y el modelo solo una vez por modelo
    model, tokenizer, device = initialize_model_and_tokenizer(model_name)

    # Para cada método de resumen
    for method in summary_methods:
        # Toma los primeros 10 ejemplos del conjunto de datos
        for i in range(0, 1000, 10):  # Procesa los ejemplos en batches de 10
            texts = [dataset['train'][j]['chapter'] for j in range(i, min(i+10, 10))]
            reference_summaries = [json.loads(dataset['train'][j]['summary'])["summary"] for j in range(i, min(i+10, 10))]

            # Genera los resúmenes utilizando el método actual
            if method == "original":
                summaries = [resumir_texto_final(["_", text], model, tokenizer, device)[1] for text in texts]
            elif method == "pipeline":
                summaries = [resumir_texto_paralelo(text, model, tokenizer, device, max_length=400, print_option="no") for text in texts]
            elif method == "original_extracted":
                summaries = [resumir_texto_final(["_", generar_resumen_extractivo(text, ratio=0.3)], model, tokenizer, device)[1] for text in texts]
            elif method == "pipeline_extracted":
                summaries = [resumir_texto_paralelo(generar_resumen_extractivo(text, ratio=0.3), model, tokenizer, device, max_length=400, print_option="no") for text in texts]

            # Calcula las puntuaciones para el método de resumen actual
            for text, reference_summary, summary in zip(texts, reference_summaries, summaries):
                scores["model_name"].append(model_name)
                scores["methods"].append(method)
                rouge = scorer.score(summary, reference_summary)
                scores["ROUGE1-Precision"].append(rouge['rouge1'].precision)
                scores["ROUGE1-Recall"].append(rouge['rouge1'].recall)
                scores["ROUGE1-F1"].append(rouge['rouge1'].fmeasure)
                scores["ROUGE2-Precision"].append(rouge['rouge2'].precision)
                scores["ROUGE2-Recall"].append(rouge['rouge2'].recall)
                scores["ROUGE2-F1"].append(rouge['rouge2'].fmeasure)
                scores["ROUGEL-Precision"].append(rouge['rougeL'].precision)
                scores["ROUGEL-Recall"].append(rouge['rougeL'].recall)
                scores["ROUGEL-F1"].append(rouge['rougeL'].fmeasure)
                scores["BLEU"].append(1)  # sentence_bleu(reference_summary, summary))
                scores["METEOR"].append(meteor_score.single_meteor_score(reference_summary.split(), summary.split()))  # Para la similitud del coseno, primero convertimos los textos a vectores
                vectorizer = CountVectorizer().fit_transform([summary, reference_summary])
                vectors = vectorizer.toarray()
                csim = cosine_similarity(vectors)
                scores["Cosine Similarity"].append(csim[0,1])  # Obtenemos la similitud del coseno del primer texto con el segundo
                # P, R, F1 = bert_score.score([summary], [reference_summary], lang='en', verbose=True)
                scores["BERTScore"].append(1)  # P.mean())
    

    
    print(model_name,method)
    df = pd.DataFrame(scores)

    # Calcula la media de las puntuaciones para cada combinación de modelo y método
    mean_scores = df.groupby(["model_name", "methods"]).mean().reset_index()

    # Imprime el DataFrame
    print(mean_scores.to_string())
    del model
    del tokenizer
    torch.cuda.empty_cache()
# Calcula las medias de las puntuaciones
df = pd.DataFrame(scores)

# Calcula la media de las puntuaciones para cada combinación de modelo y método
mean_scores = df.groupby(["model_name", "methods"]).mean().reset_index()
mean_scores.to_pickle('mean_scores.pkl')

# Lista de métricas que quieres trazar
metrics = ['ROUGE1-Precision', 'ROUGE1-Recall', 'ROUGE1-F1', 'ROUGE2-Precision', 'ROUGE2-Recall', 'ROUGE2-F1', 'ROUGEL-Precision', 'ROUGEL-Recall', 'ROUGEL-F1', 'METEOR', 'Cosine Similarity']

# Obtener los nombres de los modelos y métodos
model_names = mean_scores['model_name'].unique()
method_names = mean_scores['methods'].unique()

# Crear un color distinto para cada modelo
colors = plt.get_cmap('tab10', len(model_names))  # Usar un mapa de colores menos brillante

# Crear una figura con múltiples subplots
fig, axs = plt.subplots(4, 3, figsize=(30, 30))
axs = axs.flatten()

for j, metric in enumerate(metrics):
    ax = axs[j]
    bar_width = 0.15
    x = np.arange(len(method_names))
    bars = []  # Lista para guardar las barras
    for i, model in enumerate(model_names):
        # Filtrar los datos para cada modelo
        model_data = mean_scores[mean_scores['model_name'] == model]
        for k, method in enumerate(method_names):
            # Filtrar los datos para cada método
            method_data = model_data[model_data['methods'] == method]
            if len(method_data[metric]) > 0:  # Asegurarse de que hay datos para trazar
                bar = ax.bar(x[k] + i*bar_width, method_data[metric].values[0], color=colors(i), width=bar_width)
                if k == 0:  # Solo añadir la barra a la leyenda una vez por modelo
                    bars.append(bar)
    ax.set_xlabel('Methods')
    ax.set_ylabel(metric)
    ax.set_title(f'Bar plot of {metric} for different models and methods')
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(method_names, rotation=45)  # Rotar las etiquetas del eje x
    ax.legend(bars, model_names)  # Añadir la leyenda al gráfico de barras

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Guardar la imagen
plt.savefig('bar_plots.png')

plt.show()

# Crear una figura con múltiples subplots
fig, axs = plt.subplots(6, 2, figsize=(30, 30))
axs = axs.flatten()

for j, metric in enumerate(metrics):
    ax = axs[j]
    lines = []  # Lista para guardar las líneas
    for i, model in enumerate(model_names):
        # Filtrar los datos para cada modelo
        model_data = mean_scores[mean_scores['model_name'] == model]
        y = []
        for k, method in enumerate(method_names):
            # Filtrar los datos para cada método
            method_data = model_data[model_data['methods'] == method]
            if len(method_data[metric]) > 0:  # Asegurarse de que hay datos para trazar
                y.append(method_data[metric].values[0])
        line, = ax.plot(method_names, y, color=colors(i), marker='o')  # Dibujar una línea con marcadores
        lines.append(line)
    ax.set_xlabel('Methods')
    ax.set_ylabel(metric)
    ax.set_title(f'Line plot of {metric} for different models and methods')
    ax.legend(lines, model_names)  # Añadir la leyenda al gráfico de líneas

# Ajustar el espacio entre los subplots
plt.tight_layout()

# Guardar la imagen
plt.savefig('line_plots.png')

plt.show()