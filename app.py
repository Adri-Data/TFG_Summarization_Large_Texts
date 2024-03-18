import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import concurrent.futures
import nltk
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoTokenizer, MarianMTModel, MarianTokenizer
from pytube import YouTube
import whisperx
import threading
from deep_translator import GoogleTranslator
from summarizer import Summarizer
from queue import Queue
nltk.download('wordnet')

def initialize_model_and_tokenizer(model_name):

    # Cargar el tokenizador y el modelo
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
  summary_ids = model.generate(inputs, max_length=150, min_length=100, length_penalty=0.1, num_beams=16, no_repeat_ngram_size=4)
  return summary_ids

def transcribe_and_translate(url, source_lang='es', target_lang='en', device='cpu'):
    # Crear un objeto YouTube
    yt = YouTube(url)
    # Obtener el stream de audio de mayor calidad
    audio_stream = yt.streams.get_audio_only()
    # Descargar el audio
    audio_file = audio_stream.download()
    # Transcribir el audio usando whisperx
    model = whisperx.load_model("distil-large-v2", device, compute_type="default")
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

def start_transcription(url, source_lang, result_queue):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    translated_text = transcribe_and_translate(url, source_lang=source_lang, device=device)
    result_queue.put(translated_text)
    
def traductor(text, source='en',target='es'):
    traductor = GoogleTranslator(source=source, target=target)
    resultado = traductor.translate(text)
    return resultado


import csv

# Estilos personalizados
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

st.markdown('<p class="big-font">Resumen de Textos Largos con IA para mi TFG</p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Para mi TFG, genero resúmenes de textos largos utilizando Inteligencia Artificial.</p>', unsafe_allow_html=True)
st.markdown('Por favor, introduce la URL de un video de YouTube, selecciona el idioma del video y del resumen, y elige el modelo de IA que se utilizará para generar el resumen.')

url = st.text_input('Introduce la URL del video de YouTube', 'https://www.youtube.com/watch?v=efoTZzqOrI8')
idioma_video = st.selectbox('Selecciona el idioma del video', ['es', 'en', 'fr', 'ge'])

if st.button('Enviar URL'):
    result_queue = Queue()
    transcription_thread = threading.Thread(target=start_transcription, args=(url, idioma_video,result_queue))
    transcription_thread.start()
    st.session_state.transcription_done = False

    progress_bar = st.progress(0)
    with st.spinner('Generando Transcripción...'):
        for i in range(100):
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
    idioma_resumen = st.selectbox('Selecciona el idioma del resumen', ['es', 'en', 'fr', 'ge'])
    model_name = st.selectbox('Selecciona el modelo de IA', ['google-t5/t5-base', 'tuner007/pegasus_summarizer', 'facebook/bart-large-cnn', 'microsoft/prophetnet-large-uncased'])
    texto_procesado = False
    if st.button('Generar Resumen'):
        st.write('Generando resumen...')
        text = translated_text
        reduced_text = generar_resumen_extractivo(text, ratio=0.3)
        model, tokenizer, device = initialize_model_and_tokenizer(model_name)
        st.session_state.button_clicked = True  
        

        _, summary_original = resumir_texto_final([0, translated_text], model, tokenizer, device)
        st.write(f"Model: {model_name}")
        st.write("_________________________________________________________________\n\n")
        st.write(f"\n Generated Summary without the pipeline: {summary_original}")
        st.write(f"\n Generated Summary without the pipeline: {traductor(summary_original)}")
    
        summary_pipeline = resumir_texto_paralelo(text, model, tokenizer, device, max_length=400, print_option="no")
        st.write(f"\n Generated Summary with the pipeline: {summary_pipeline}")
        st.write(f"\n Generated Summary with the pipeline: {traductor(summary_pipeline)}")
    
        _, summary_original_extracted = resumir_texto_final([0, reduced_text], model, tokenizer, device)
        st.write(f"\n Generated Summary with extractive summarization: {summary_original_extracted}")
        st.write(f"\n Generated Summary with extractive summarization: {traductor(summary_original_extracted)}")
    
        summary_pipeline_extracted = resumir_texto_paralelo(reduced_text, model, tokenizer, device, max_length=400, print_option="no")
        st.write(f"\n Generated Summary with pipeline and extractive summarization: {summary_pipeline_extracted}")
        st.write(f"\n Generated Summary with pipeline and extractive summarization: {traductor(summary_pipeline_extracted)}")
    
        st.write("_________________________________________________________________\n\n")
        

        # Sistema de feedback
        summary_options = [summary_original, summary_pipeline, summary_original_extracted, summary_pipeline_extracted]
        st.session_state.summary_options = summary_options
    if 'button_clicked' in st.session_state:
        summary_options = st.session_state.summary_options
        summary_labels = ["Resumen original", "Resumen con pipeline", "Resumen con resumen extractivo", "Resumen con pipeline y resumen extractivo"]
        st.write(f"Model: {model_name}")
        st.write("_________________________________________________________________\n\n")
        st.write(f"\n Generated Summary without the pipeline: {summary_original}")
        st.write(f"\n Generated Summary without the pipeline: {traductor(summary_original)}")
    
        st.write(f"\n Generated Summary with the pipeline: {summary_pipeline}")
        st.write(f"\n Generated Summary with the pipeline: {traductor(summary_pipeline)}")
    
        st.write(f"\n Generated Summary with extractive summarization: {summary_original_extracted}")
        st.write(f"\n Generated Summary with extractive summarization: {traductor(summary_original_extracted)}")
    
        st.write(f"\n Generated Summary with pipeline and extractive summarization: {summary_pipeline_extracted}")
        st.write(f"\n Generated Summary with pipeline and extractive summarization: {traductor(summary_pipeline_extracted)}")
    
        st.write("_________________________________________________________________\n\n")
        # Crea dos columnas
        col1, col2 = st.columns(2)
        
        # Coloca el widget de selección en la primera columna y el botón en la segunda
        with col1:
            user_vote = st.radio("Selecciona tu resumen favorito:", options=range(len(summary_options)), format_func=lambda x: summary_labels[x])
        with col2:
            if st.button('Enviar voto'):
                st.write(f"Has votado por: {summary_labels[user_vote]}")
                
                # Almacenamiento de votos
                with open('votes.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([summary_labels[user_vote]])

