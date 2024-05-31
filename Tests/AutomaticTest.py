# Importar las librerías necesarias
#%pip install pytube
#%pip install rouge_score
#%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#%pip install git+https://github.com/m-bain/whisperx.git --upgrade
#%pip install datasets
#%pip install sentencepiece
#%pip install accelerate -U
#%pip install bert-extractive-summarizer
#%pip install bert-score

import concurrent.futures
import bert_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import pandas as pd
from summarizer import Summarizer
from datasets import load_dataset
import sentencepiece
from accelerate import Accelerator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import json
from pytube import YouTube
import whisperx
import gc
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import MarianMTModel, MarianTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import nltk
import re
from nltk.translate import meteor_score
import nltk
#nltk.download('wordnet')


def initialize_model_and_tokenizer(model_name):
    # Cargar el tokenizador y el modelo
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

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

    # Crea un pool de trabajadores para ejecutar resumir_texto en paralelo
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

# Define los modelos que quieres probar
model_names = ["tuner007/pegasus_summarizer", "facebook/bart-large-cnn", "t5-base", "microsoft/prophetnet-large-uncased"]

# Carga el conjunto de datos booksum
dataset = load_dataset('kmfoda/booksum')

# Inicializa el evaluador de ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Para cada modelo
for model_name in model_names:
    # Carga el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if (model_name=="tuner007/pegasus_summarizer" or model_name=="microsoft/prophetnet-large-uncased"):
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
      model = AutoModelWithLMHead.from_pretrained(model_name)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Solo toma el primer ejemplo del conjunto de datos
    example = dataset['train'][0]

    # Define el texto que quieres resumir
    text = example['chapter']
    dictionary = json.loads(example['summary'])
    reference_summary = dictionary["summary"]
    reduced_text = generar_resumen_extractivo(text, ratio=0.3)

    #resumen 1 model
    inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = sumary_final(inputs)
    summary_original = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    #resume pipeline
    summary_pipeline = resumir_texto_paralelo(text, model, tokenizer, device, max_length=400, print_option="no")

    #resumen 1 model extractive
    inputs = tokenizer.encode("summarize: " + reduced_text, return_tensors='pt', max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = sumary_final(inputs)
    summary_original_extracted = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    #resume pipeline
    summary_pipeline_extracted = resumir_texto_paralelo(reduced_text, model, tokenizer, device, max_length=400, print_option="no")


    # Calcula las puntuaciones de ROUGE
    scores_pipeline = scorer.score(summary_pipeline,reference_summary)
    scores_original = scorer.score(summary_original,reference_summary)
    scores_original_extracted = scorer.score(summary_original_extracted,reference_summary)
    scores_pipeline_extracted = scorer.score(summary_pipeline_extracted,reference_summary)

    # Calcula las puntuaciones
    scores = {
        "model_name": [],
        "Model": [],
        "ROUGE1-Precision": [],
        "ROUGE1-F1": [],
        "ROUGE2-Precision": [],
        "ROUGE2-F1": [],
        "ROUGEL-Precision": [],
        "ROUGEL-F1": [],
        "BLEU": [],
        "METEOR": [],
        "Cosine Similarity": [],
        "BERTScore": [],
    }

    summaries = [summary_original, summary_pipeline, summary_original_extracted, summary_pipeline_extracted]
    names = ["Original", "Pipeline", "Original Extracted", "Pipeline Extracted"]

    for name, summary in zip(names, summaries):
        scores["model_name"].append(model_name)
        scores["Model"].append(name)
        rouge = scorer.score(summary, reference_summary)
        scores["ROUGE1-Precision"].append(rouge['rouge1'].precision)  # Añade la precisión de ROUGE1
        scores["ROUGE1-F1"].append(rouge['rouge1'].fmeasure)  # Añade la puntuación F1 de ROUGE1
        scores["ROUGE2-Precision"].append(rouge['rouge2'].precision)  # Añade la precisión de ROUGE2
        scores["ROUGE2-F1"].append(rouge['rouge2'].fmeasure)  # Añade la puntuación F1 de ROUGE2
        scores["ROUGEL-Precision"].append(rouge['rougeL'].precision)  # Añade la precisión de ROUGEL
        scores["ROUGEL-F1"].append(rouge['rougeL'].fmeasure)  # Añade la puntuación F1 de ROUGEL
        scores["BLEU"].append(1)#sentence_bleu(reference_summary, summary))
        #print(reference_summary.split(), summary.split())
        scores["METEOR"].append(meteor_score.single_meteor_score(reference_summary.split(), summary.split())) # Para la similitud del coseno, primero convertimos los textos a vectores
        vectorizer = CountVectorizer().fit_transform([summary, reference_summary])
        vectors = vectorizer.toarray()
        csim = cosine_similarity(vectors)
        scores["Cosine Similarity"].append(csim[0,1])  # Obtenemos la similitud del coseno del primer texto con el segundo

        # BERTScore
        P, R, F1 = bert_score.score([summary], [reference_summary], lang='en', verbose=True)
        print(f"BERTScore: P={P.mean()}, R={R.mean()}, F1={F1.mean()}")
        scores["BERTScore"].append(P.mean())
    # Presenta los resultados en una tabla
    df = pd.DataFrame(scores)
    print(df.to_string())


    print(f"Model: {model_name}")

    print(f"Reference Summary: {reference_summary}")
    print("_________________________________________________________________\n\n")

    print(f"\n Generated Summary without the pipeline: {summary_original}")

    print(f"\n Generated Summary with the pipeline: {summary_pipeline}")

    print(f"\n Generated Summary with extractive summarization: {summary_original_extracted}")

    print(f"\n Generated Summary with pipeline and extractive summarization: {summary_pipeline_extracted}")

    print("_________________________________________________________________\n\n")
