import streamlit as st
import numpy as np
import concurrent.futures
import threading
import requests
import whisperx
import nltk
import torch
import time
import random
import csv

from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoTokenizer, MarianMTModel, MarianTokenizer
from concurrent.futures import ThreadPoolExecutor
from deep_translator import GoogleTranslator
from streamlit_lottie import st_lottie
from summarizer import Summarizer
from pytube import YouTube
from queue import Queue
from PIL import Image

nltk.download('wordnet')

def initialize_model_and_tokenizer(model_name):

    # Cargar el tokenizador y el modelo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if (model_name=="tuner007/pegasus_summarizer" or model_name=="microsoft/prophetnet-large-uncased"):
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
      model = AutoModelWithLMHead.from_pretrained(model_name)

    # Mover el modelo a la GPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, tokenizer, device

def sumary_final(inputs):
  
  summary_ids = model.generate(inputs, max_length=150, min_length=100, length_penalty=0.1, num_beams=16, no_repeat_ngram_size=4)
  return summary_ids

def transcribe_and_translate(url, source_lang='es', target_lang='en', device='cpu'):
    # Buscar el video en youtube
    yt = YouTube(url)
    # Obtener audio 
    audio_stream = yt.streams.get_audio_only()
    # Descargar el audio
    audio_file = audio_stream.download()
    # Transcribir el audio usando whisperx
    model = whisperx.load_model("distil-large-v2", device, compute_type="default")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)
    
    model_a, metadata = whisperx.load_align_model(language_code=source_lang, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # Traducir el texto al inglés
    if (source_lang!=target_lang):
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

        with ThreadPoolExecutor() as executor:
            segmentos = [(i, diccionario['text']) for i, diccionario in enumerate(result["segments"])]
            traducciones = list(executor.map(traducir_segmento, segmentos))
            # Ordenar las traducciones por id y juntarlas
            texto_traducido = " ".join(traduccion for id, traduccion in sorted(traducciones))
    else:
        #No hay cambio de idioma
        texto_traducido = " ".join(segmento['text'] for segmento in result["segments"])

    return texto_traducido


def divide_texto(texto, tokenizer, longitud_max=512):

    nltk.download('punkt', quiet=True)

    # Divide el texto en oraciones con NLTK
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
    # Si el número de tokens es menor que max_length, llama a resumir_texto_final
    if len(tokenizer.encode(texto)) <= max_length:
        return resumir_texto_final((None, texto), model, tokenizer, device)[1]


    trozos = divide_texto(texto, tokenizer, max_length)
    resumenes = {}

    # Crea un pool para ejecutar resumir_texto en paralelo
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

def start_transcription(url, source_lang, result_queue):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    translated_text = transcribe_and_translate(url, source_lang=source_lang, device=device)
    result_queue.put(translated_text)
    

def traductor(text, source='en',target='es'):
    traductor = GoogleTranslator(source=source, target=target)
    resultado = traductor.translate(text)
    return resultado




#Comienza la pagina de Streamlit
st.markdown("""
<style>
    .reportview-container {
        background: #f0f0f5
    }
    .big-font {
        font-size:50px !important;
    }
    .medium-font {
        font-size:25px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="medium-font">TFG🎓| Resumidor de videos con IA🤖</p>', unsafe_allow_html=True)
st.markdown('Para mi TFG, genero resúmenes a partir de transcripciones de videos con modelos de Inteligencia Artificial.')

ID = np.random.randint(1, 1e6)
url = st.text_input('Cambia la URL al video de YouTube que quieras⬇️', 'https://www.youtube.com/watch?v=efoTZzqOrI8')
st.video(url)
idioma_video = st.selectbox('¿En qué idioma esta el video?🔠', ['es', 'en', 'fr', 'ge'])

if st.button('Enviar URL'):
    result_queue = Queue()
    transcription_thread = threading.Thread(target=start_transcription, args=(url, idioma_video ,result_queue))
    transcription_thread.start()
    st.session_state.transcription_done = False

    progress_bar = st.progress(0)
    with st.spinner('Generando Transcripción...✍️'):
        for i in range(95):
            if not transcription_thread.is_alive():
                progress_bar.progress(100)
                break
            else:
                time.sleep(2)
                progress_bar.progress(i + 1)

    if transcription_thread.is_alive():
        transcription_thread.join()
    translated_text = result_queue.get()
    st.session_state.transcription_done = True
    st.session_state.translated_text = translated_text

if 'transcription_done' in st.session_state and st.session_state.transcription_done:
    translated_text = st.session_state.translated_text
    idioma_resumen = st.selectbox('Selecciona el idioma del resumen🔠', ['es', 'en', 'fr', 'ge'])
    model_name = st.selectbox('Selecciona el modelo de IA🤖', ['facebook/bart-large-cnn', 'google-t5/t5-base', 'microsoft/prophetnet-large-uncased', 'tuner007/pegasus_summarizer'])
    texto_procesado = False
    if st.button('Generar Resumen🗒️'):
        st.session_state.texto_procesado = False
        progress_bar = st.progress(0)
        with st.spinner('Preparando para generar el resumen...'):
            text = translated_text
            
            reduced_text = generar_resumen_extractivo(text, ratio=0.3)
        with st.spinner('Texto reducido listo...'):
            progress_bar.progress(20)
            
            model, tokenizer, device = initialize_model_and_tokenizer(model_name)
        with st.spinner('Modelo inicializado...'):
            progress_bar.progress(40)
    
            _, summary_original = resumir_texto_final([0, translated_text], model, tokenizer, device)
        with st.spinner('Resumen generado sin la pipeline...'):
            progress_bar.progress(60)
    
            summary_pipeline = resumir_texto_paralelo(text, model, tokenizer, device, max_length=400, print_option="no")
        with st.spinner('Resumen generado con la pipeline...'):
            progress_bar.progress(80)
    
            _, summary_original_extracted = resumir_texto_final([0, reduced_text], model, tokenizer, device)
        with st.spinner('Resumen generado con la sumarización extractiva...'):
            progress_bar.progress(90)
    
            summary_pipeline_extracted = resumir_texto_paralelo(reduced_text, model, tokenizer, device, max_length=400, print_option="no")
        with st.spinner('Resumen generado con la pipeline y la sumarización extractiva...'):
            progress_bar.progress(100)
    
            st.success('¡Resumen generado con éxito!')
            st.session_state.texto_procesado = True
            st.session_state.button_clicked = True
            # Sistema de feedback
            summary_options = [summary_original, summary_pipeline, summary_original_extracted, summary_pipeline_extracted]
            summary_labels = ["Resumen original", "Resumen con pipeline", "Resumen con resumen extractivo", "Resumen con pipeline y resumen extractivo"]
            
            # Se mezcla para que no se sepa cual es cual
            combined = list(zip(summary_labels, summary_options))
            random.shuffle(combined)
            summary_labels_shuffled, summary_options_shuffled = zip(*combined)

            st.session_state.summary_options = summary_options_shuffled
            st.session_state.summary_labels = summary_labels_shuffled
            st.session_state.first_time = True

if 'button_clicked' in st.session_state:
    summary_options = st.session_state.summary_options
    summary_labels = st.session_state.summary_labels

    st.write(f"Model: {model_name}")
    st.write("_________________________________________________________________\n\n")
    st.write("Aquí hay 4 resumenes generados con distintos metodos, por favor valora cada resumen con una puntuación del 1(malo)❌ al 5(bueno)✅\n\n")
    st.write("_________________________________________________________________\n\n")
    
    anonymous_labels = [f"Resumen {chr(i)}" for i in range(65, 65 + len(summary_options))] 
    
    # Inicializa las puntuaciones si aún no existen
    if 'scores' not in st.session_state:
        st.session_state.scores = [1] * len(summary_options)
    
    for i, (label, option) in enumerate(zip(anonymous_labels, summary_options)):
        if idioma_resumen == "en":
            st.write(f"\n {label}: {option}")
        else:
            st.write(f"\n {label}: {traductor(option)}")
        st.session_state.scores[i] = st.radio(f"Puntuación para {label}🏆:", options=[1, 2, 3, 4, 5], index=st.session_state.scores[i]-1)
    
    if st.button('Enviar votos🗳️'):
        with open('votes.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            for label, score, option in zip(summary_labels, st.session_state.scores, summary_options):
                writer.writerow([ID, url, idioma_video, model_name, label, score, option])
        st.write("Tus votos han sido registrados. ¡Gracias!")
