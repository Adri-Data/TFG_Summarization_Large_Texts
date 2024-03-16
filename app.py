import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image
import concurrent.futures
import nltk
import torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoTokenizer, MarianMTModel, MarianTokenizer
from pytube import YouTube
import whisperx
import streamlit as st
import requests
import streamlit_lottie as st_lottie
import threading
from deep_translator import GoogleTranslator
from summarizer import Summarizer
nltk.download('wordnet')

def initialize_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name in ["tuner007/pegasus_summarizer", "microsoft/prophetnet-large-uncased"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelWithLMHead.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, tokenizer, device

def transcribe_and_translate(url, source_lang='es', target_lang='en', device='cuda'):
    yt = YouTube(url)
    audio_stream = yt.streams.get_audio_only()
    audio_file = audio_stream.download()
    model = whisperx.load_model("large-v2", device, compute_type="float16")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=16)
    model_a, metadata = whisperx.load_align_model(language_code=source_lang, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    modelo_nombre = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    modelo = MarianMTModel.from_pretrained(modelo_nombre).to(device)
    tokenizador = MarianTokenizer.from_pretrained(modelo_nombre)
    def traducir_segmento(segmento):
        id, texto = segmento
        texto_tokenizado = tokenizador.prepare_seq2seq_batch([texto], return_tensors='pt').to(device)
        traduccion_ids = modelo.generate(**texto_tokenizado)
        traduccion = tokenizador.decode(traduccion_ids[0], skip_special_tokens=True)
        return id, traduccion
    with ThreadPoolExecutor() as executor:
        segmentos = [(i, diccionario['text']) for i, diccionario in enumerate(result["segments"])]
        traducciones = list(executor.map(traducir_segmento, segmentos))
    texto_traducido = " ".join(traduccion for id, traduccion in sorted(traducciones))
    return texto_traducido

def divide_texto(texto, tokenizer, longitud_max=512):
    nltk.download('punkt', quiet=True)
    oraciones = nltk.tokenize.sent_tokenize(texto)
    trozos = []
    trozo_actual = ''
    for i, parrafo in enumerate(oraciones):
        if len(tokenizer.encode(trozo_actual + ' \n ' + parrafo)) <= longitud_max:
            trozo_actual += '\n' + parrafo
        else:
            trozos.append((i, trozo_actual))
            trozo_actual = parrafo
    trozos.append((i, trozo_actual))
    return trozos

def resumir_texto(trozo, model, tokenizer, device):
    id, texto = trozo
    inputs = tokenizer.encode("summarize: " + texto, return_tensors='pt', max_length=512, truncation=True, padding='longest').to(device)
    summary_ids = model.generate(inputs, max_length=100, min_length=20, length_penalty=3, num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
    resumen = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return id, resumen


def resumir_texto_paralelo(texto, model, tokenizer, device, max_length):
    if len(tokenizer.encode(texto)) <= max_length:
        return resumir_texto((None, texto), model, tokenizer, device)[1]
    trozos = divide_texto(texto, tokenizer, max_length)
    resumenes = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_trozo = {executor.submit(resumir_texto, trozo, model, tokenizer, device): trozo for trozo in trozos}
        for i, future in enumerate(concurrent.futures.as_completed(future_to_trozo)):
            trozo = future_to_trozo[future]
            try:
                id, resumen = future.result()
            except Exception as exc:
                print(f'El trozo {trozo} generó una excepción: {exc}')
            else:
                resumenes[id] = resumen
    texto_resumen = ' '.join(resumen for id, resumen in sorted(resumenes.items()))
    texto_resumen = resumir_texto_paralelo(texto_resumen, model, tokenizer, device, max_length)
    return texto_resumen

def generar_resumen_extractivo(texto, ratio, algoritmo="pagerank"):
    model = Summarizer()
    resumen = model(texto, ratio=ratio, algorithm=algoritmo)
    return resumen

def start_transcription(url, source_lang):
    global translated_text
    translated_text = transcribe_and_translate(url, source_lang=source_lang)

def traductor(text, source='en',target='es'):
    traductor = GoogleTranslator(source=source, target=target)
    resultado = traductor.translate(text)
    return resultado


st.title('Resumen de Textos Largos con IA para mi TFG')
st.write('''
Para mi TFG, genero resúmenes de textos largos utilizando Inteligencia Artificial.
Por favor, introduce la URL de un video de YouTube, selecciona el idioma del video y del resumen,
y elige el modelo de IA que se utilizará para generar el resumen.
''')
url = st.text_input('Introduce la URL del video de YouTube')
idioma_video = st.selectbox('Selecciona el idioma del video', ['es', 'en', 'fr', 'ge'])
if st.button('Enviar URL'):
    transcription_thread = threading.Thread(target=start_transcription, args=(url, idioma_video))
    transcription_thread.start()
    idioma_resumen = st.selectbox('Selecciona el idioma del resumen', ['es', 'en', 'fr', 'ge'])
    model_name = st.selectbox('Selecciona el modelo de IA', ['google-t5/t5-base', 'tuner007/pegasus_summarizer', 'facebook/bart-large-cnn', 'microsoft/prophetnet-large-uncased'])
    if st.button('Generar Resumen'):
        st.write('Generando resumen...')
        text = translated_text
        reduced_text = generar_resumen_extractivo(text, ratio=0.3)
        model, tokenizer, device = initialize_model_and_tokenizer(model_name)
        _, summary_original = resumir_texto_final([_, translated_text], model, tokenizer, device)
        st.write(f"Model: {model_name}")
        st.write("_________________________________________________________________\n\n")
        st.write(f"\n Generated Summary without the pipeline: {summary_original}")
        st.write(f"\n Generated Summary without the pipeline: {traductor(summary_original)}")
    
        summary_pipeline = resumir_texto_paralelo(text, model, tokenizer, device, max_length=400, print_option="no")
        st.write(f"\n Generated Summary with the pipeline: {summary_pipeline}")
        st.write(f"\n Generated Summary with the pipeline: {traductor(summary_pipeline)}")
    
        _, summary_original_extracted = resumir_texto_final([_, reduced_text], model, tokenizer, device)
        st.write(f"\n Generated Summary with extractive summarization: {summary_original_extracted}")
        st.write(f"\n Generated Summary with extractive summarization: {traductor(summary_original_extracted)}")
    
        summary_pipeline_extracted = resumir_texto_paralelo(reduced_text, model, tokenizer, device, max_length=400, print_option="no")
        st.write(f"\n Generated Summary with pipeline and extractive summarization: {summary_pipeline_extracted}")
        st.write(f"\n Generated Summary with pipeline and extractive summarization: {traductor(summary_pipeline_extracted)}")
    
        st.write("_________________________________________________________________\n\n")
