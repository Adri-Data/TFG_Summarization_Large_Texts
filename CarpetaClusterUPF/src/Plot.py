import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos
mean_scores = pd.read_pickle('mean_scores.pkl')

# Lista de métricas que quieres trazar
metrics = ['ROUGE1-Precision', 'ROUGE1-Recall', 'ROUGE1-F1', 'ROUGE2-Precision', 'ROUGE2-Recall', 'ROUGE2-F1', 'ROUGEL-Precision', 'ROUGEL-Recall', 'ROUGEL-F1', 'BLEU', 'METEOR', 'Cosine Similarity']

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
