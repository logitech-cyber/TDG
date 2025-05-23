import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from huggingface_hub import hf_hub_download

# --- Configuración de Hugging Face Hub ---
# ¡IMPORTANTE!: Reemplaza "tu_usuario/nombre_de_tu_repo_modelos" con el ID real de tu repositorio en Hugging Face
HF_REPO_ID = "branddiego/model_valence_arousal" 

# Nombres de los archivos de los modelos en el repositorio de Hugging Face
MODEL_AROUSAL_FILENAME = "modelo_arousal_random_forest_todas_caracteristicas.joblib"
MODEL_VALENCE_FILENAME = "modelo_valence_random_forest_todas_caracteristicas.joblib"

# --- Función para cargar un único modelo desde Hugging Face Hub ---
# Usamos st.cache_data aquí porque queremos que el modelo se almacene en caché una vez
# que se carga, pero podemos controlar cuándo se llama.
@st.cache_data(show_spinner=False) # show_spinner=False para controlar el spinner manualmente
def load_single_model_from_hf(repo_id, filename, token=None):
    """
    Descarga un único modelo (o un componente joblib) desde Hugging Face Hub y lo carga.
    Utiliza st.cache_data para asegurar que la descarga y carga solo ocurra una vez por modelo.
    """
    st.info(f"Cargando {filename} desde Hugging Face Hub...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=None # Esto es útil si tu repo HF es privado
        )
        componentes = joblib.load(model_path)
        st.success(f"{filename} cargado con éxito.")
        return componentes
    except Exception as e:
        st.error(f"Error al descargar o cargar el modelo {filename} desde Hugging Face Hub: {e}")
        st.warning("Asegúrate de que el HF_REPO_ID es correcto y los modelos existen en el repositorio. Si es un repo privado, verifica el token.")
        st.stop() # Detiene la ejecución de la app si no se pueden cargar los modelos

# NO cargar modelos al inicio del script. Se cargarán bajo demanda.
# arousal_model_components = load_model_from_hf(...)
# valence_model_components = load_model_from_hf(...)


# --- Función para predecir emociones (adaptada) ---
def predecir_emocion_desde_csv(df, componentes_modelo, columna_id=None):
    """
    Predice arousal/valence a partir de un DataFrame con características extraídas
    utilizando componentes del modelo directamente (modelo, scaler, caracteristicas).
    """
    modelo = componentes_modelo['modelo']
    scaler = componentes_modelo['scaler']
    caracteristicas_modelo = componentes_modelo['caracteristicas']

    st.write("Preparando características para predicción...")
    # Verificar si todas las características necesarias están presentes
    caracteristicas_faltantes = set(caracteristicas_modelo) - set(df.columns)
    if caracteristicas_faltantes:
        st.warning(f"Faltan {len(caracteristicas_faltantes)} características en el CSV.")
        if len(caracteristicas_faltantes) <= 5:
            st.warning(f"Características faltantes: {caracteristicas_faltantes}")
        else:
            st.warning(f"Primeras 5 faltantes: {list(caracteristicas_faltantes)[:5]}")
        
        # Rellenar con ceros si faltan características
        for caract in caracteristicas_faltantes:
            st.warning(f"La característica '{caract}' no se encontró en el CSV. Se rellenará con ceros.")
            df[caract] = np.zeros(len(df))

    # Crear DataFrame con características en el orden correcto
    X_pred = df[caracteristicas_modelo] # Ahora que rellenamos df, podemos hacer esto directamente
    
    # Escalar características
    X_scaled = scaler.transform(X_pred)

    st.write("Realizando predicciones...")
    predicciones = modelo.predict(X_scaled)

    # Crear DataFrame de resultados
    if columna_id is not None and columna_id in df.columns:
        resultados = pd.DataFrame({
            'ID': df[columna_id],
            'prediccion': predicciones
        })
    else:
        resultados = pd.DataFrame({
            'prediccion': predicciones
        })
    
    return resultados


# Título y descripción de la aplicación
st.title("Predictor de Emociones Musicales")
st.markdown("Utiliza modelos de machine learning para predecir arousal y valence en muestras musicales")

# Crear secciones para la aplicación
st.header("1. Cargar archivo CSV con características")

# Opción para cargar archivo
uploaded_file = st.file_uploader("Selecciona un archivo CSV con características", type=["csv"])

if uploaded_file is not None:
    # Mostrar proceso de carga
    with st.spinner('Cargando y procesando archivo...'):
        # Leer el archivo cargado
        df_original = pd.read_csv(uploaded_file, delimiter=';') # Guardamos el original
        df = df_original.copy() # Trabajamos con una copia para evitar modificar el original
        
        # Mostrar información del archivo
        st.success(f"Archivo cargado correctamente: {uploaded_file.name}")
        st.write(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        
        # Mostrar primeras filas
        with st.expander("Ver primeras filas del CSV"):
            st.dataframe(df.head())
        
        # Identificar posible columna ID
        columna_id = None
        for posible_id in ['song_id', 'sample', 'ID', 'id', 'filename']:
            if posible_id in df.columns:
                columna_id = posible_id
                st.info(f"Columna de identificación detectada: {columna_id}")
                break
        
        if columna_id is None:
            st.warning("No se detectó una columna de identificación. Las predicciones se numerarán secuencialmente.")
    
    # Sección para seleccionar el tipo de predicción
    st.header("2. Realizar Predicciones")
    
    # Usamos st.session_state para mantener los resultados entre ejecuciones
    if 'resultados_arousal' not in st.session_state:
        st.session_state['resultados_arousal'] = None
    if 'resultados_valence' not in st.session_state:
        st.session_state['resultados_valence'] = None
    if 'resultados_combinados' not in st.session_state:
        st.session_state['resultados_combinados'] = None

    # Opciones de botones para cada predicción
    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        if st.button("Predecir Arousal"):
            st.subheader("Predicción de Arousal")
            with st.spinner("Cargando modelo de Arousal y realizando predicciones..."):
                # Cargar el modelo de Arousal solo cuando se presiona el botón
                arousal_model_components = load_single_model_from_hf(
                    HF_REPO_ID, 
                    MODEL_AROUSAL_FILENAME, 
                    token=None
                )
                
                # Realizar predicción de Arousal
                resultados_arousal_temp = predecir_emocion_desde_csv(
                    df.copy(), # Pasamos una copia para evitar side-effects
                    arousal_model_components,
                    columna_id
                )
                resultados_arousal_temp.rename(columns={'prediccion': 'arousal'}, inplace=True)
                st.session_state['resultados_arousal'] = resultados_arousal_temp
                st.success("Predicción de Arousal completada.")
                st.dataframe(st.session_state['resultados_arousal']) # Mostrar resultados parciales

            # Liberar explícitamente el modelo de arousal si es posible (Python GC lo manejará)
            # del arousal_model_components # Esto intenta liberar la memoria, pero st.cache_data podría mantenerla.
            # import gc; gc.collect() # Fuerza la recolección de basura, pero no garantiza liberación inmediata de caché.

    with col_pred2:
        if st.button("Predecir Valence"):
            st.subheader("Predicción de Valence")
            with st.spinner("Cargando modelo de Valence y realizando predicciones..."):
                # Cargar el modelo de Valence solo cuando se presiona el botón
                valence_model_components = load_single_model_from_hf(
                    HF_REPO_ID, 
                    MODEL_VALENCE_FILENAME, 
                    token=None
                )

                # Realizar predicción de Valence
                resultados_valence_temp = predecir_emocion_desde_csv(
                    df.copy(), # Pasamos una copia para evitar side-effects
                    valence_model_components,
                    columna_id
                )
                resultados_valence_temp.rename(columns={'prediccion': 'valence'}, inplace=True)
                st.session_state['resultados_valence'] = resultados_valence_temp
                st.success("Predicción de Valence completada.")
                st.dataframe(st.session_state['resultados_valence']) # Mostrar resultados parciales
            
            # del valence_model_components
            # import gc; gc.collect()

    # Si ambos resultados están disponibles, combinarlos y mostrar la visualización
    if st.session_state['resultados_arousal'] is not None and st.session_state['resultados_valence'] is not None:
        st.header("3. Resultados Combinados y Visualización")
        if columna_id:
            st.session_state['resultados_combinados'] = st.session_state['resultados_arousal'].merge(
                st.session_state['resultados_valence'], on='ID'
            )
        else:
            # Si no hay ID, asumir mismo orden y combinar
            st.session_state['resultados_combinados'] = st.session_state['resultados_arousal'].copy()
            st.session_state['resultados_combinados']['valence'] = st.session_state['resultados_valence']['valence']

        # Mostrar estadísticas
        st.subheader("Estadísticas de predicciones")
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.metric("Media de Arousal", f"{st.session_state['resultados_combinados']['arousal'].mean():.4f}")
            st.metric("Mínimo de Arousal", f"{st.session_state['resultados_combinados']['arousal'].min():.4f}")
            st.metric("Máximo de Arousal", f"{st.session_state['resultados_combinados']['arousal'].max():.4f}")
        
        with col_stats2:
            st.metric("Media de Valence", f"{st.session_state['resultados_combinados']['valence'].mean():.4f}")
            st.metric("Mínimo de Valence", f"{st.session_state['resultados_combinados']['valence'].min():.4f}")
            st.metric("Máximo de Valence", f"{st.session_state['resultados_combinados']['valence'].max():.4f}")
        
        # Mostrar resultados en tabla
        st.subheader("Tabla de Predicciones Combinadas")
        st.dataframe(st.session_state['resultados_combinados'])
        
        # Visualización
        st.subheader("Visualización de Predicciones")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='valence', 
            y='arousal', 
            data=st.session_state['resultados_combinados'],
            alpha=0.7,
            ax=ax
        )
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Valence', fontsize=12)
        plt.ylabel('Arousal', fontsize=12)
        plt.title('Mapa Emocional: Arousal vs Valence', fontsize=14)
        
        plt.text(0.25, 0.75, 'Tenso', fontsize=12, ha='center')
        plt.text(0.75, 0.75, 'Alegre', fontsize=12, ha='center')
        plt.text(0.25, 0.25, 'Triste', fontsize=12, ha='center')
        plt.text(0.75, 0.25, 'Calmado', fontsize=12, ha='center')
        
        st.pyplot(fig)
        
        # Opción para descargar resultados
        csv = st.session_state['resultados_combinados'].to_csv(index=False)
        st.download_button(
            label="Descargar predicciones como CSV",
            data=csv,
            file_name="predicciones_emocionales.csv",
            mime="text/csv",
        )
    else:
        st.info("Presiona 'Predecir Arousal' y luego 'Predecir Valence' para ver los resultados combinados y la visualización.")

# Agregar información adicional
with st.sidebar:
    st.title("Información")
    st.markdown("""
    ### Sobre esta aplicación
    
    Esta herramienta permite predecir las dimensiones emocionales de **arousal** y **valence** en muestras musicales utilizando modelos de machine learning previamente entrenados.
    
    - **Arousal**: Nivel de activación o energía (alto vs. bajo)
    - **Valence**: Nivel de positividad o negatividad (positivo vs. negativo)
    
    ### Instrucciones
    
    1. Sube un archivo CSV con las características extraídas de tus muestras musicales.
    2. Haz clic en "Predecir Arousal".
    3. Luego, haz clic en "Predecir Valence".
    4. Explora los resultados y descarga las predicciones combinadas.
    
    ### Requisitos del CSV
    
    El CSV debe contener columnas con las características necesarias para los modelos. Opcionalmente, puede incluir una columna de identificación ('song_id', 'sample', 'ID', 'filename').
    """)
    
    st.markdown("---")
    st.markdown("Desarrollado para análisis de emociones musicales")
