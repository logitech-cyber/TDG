import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from huggingface_hub import hf_hub_download # Importa hf_hub_download

# --- Configuración de Hugging Face Hub ---
# ¡IMPORTANTE!: Reemplaza "tu_usuario/nombre_de_tu_repo_modelos" con el ID real de tu repositorio en Hugging Face
HF_REPO_ID = "branddiego/model_valence_arousal" 

# Nombres de los archivos de los modelos en el repositorio de Hugging Face
MODEL_AROUSAL_FILENAME = "modelo_arousal_random_forest_todas_caracteristicas.joblib"
MODEL_VALENCE_FILENAME = "modelo_valence_random_forest_todas_caracteristicas.joblib"

# --- Función para cargar modelos desde Hugging Face Hub ---
@st.cache_resource
def load_model_from_hf(repo_id, filename, token=None):
    """
    Descarga un modelo (o un componente joblib) desde Hugging Face Hub y lo carga.
    Utiliza st.cache_resource para asegurar que la descarga y carga solo ocurra una vez.
    """
    st.info(f"Cargando {filename} desde Hugging Face Hub (se hará solo una vez)...")
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

# --- Cargar modelos al inicio de la aplicación ---
# Usamos st.secrets.get("HF_TOKEN") para obtener el token de Hugging Face de forma segura,
# si tu repositorio es privado. Si es público, puedes pasar token=None.
arousal_model_components = load_model_from_hf(
    HF_REPO_ID, 
    MODEL_AROUSAL_FILENAME, 
    token=None # No pases un token si el repositorio es público
)
valence_model_components = load_model_from_hf(
    HF_REPO_ID, 
    MODEL_VALENCE_FILENAME, 
    token=None # No pases un token si el repositorio es público
)

# Título y descripción de la aplicación
st.title("Predictor de Emociones Musicales")
st.markdown("Utiliza modelos de machine learning para predecir arousal y valence en muestras musicales")

# La función `predecir_emocion_desde_csv` ya no necesita la `ruta_modelo`,
# sino los componentes del modelo directamente.
def predecir_emocion_desde_csv(df, componentes_modelo, columna_id=None):
    """
    Predice arousal/valence a partir de un DataFrame con características extraídas
    utilizando componentes del modelo directamente (modelo, scaler, caracteristicas).

    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con las características extraídas
    componentes_modelo : dict
        Diccionario que contiene 'modelo', 'scaler' y 'caracteristicas' del modelo.
    columna_id : str, opcional
        Nombre de la columna que contiene identificadores (si existe)

    Retorna:
    --------
    pandas.DataFrame: Predicciones con identificadores (si se proporcionaron)
    """
    # 1. Extraer el modelo y componentes del diccionario
    modelo = componentes_modelo['modelo']
    scaler = componentes_modelo['scaler']
    caracteristicas_modelo = componentes_modelo['caracteristicas']

    with st.spinner("Preparando características para predicción..."):
        # Verificar si todas las características necesarias están presentes
        caracteristicas_faltantes = set(caracteristicas_modelo) - set(df.columns)
        if caracteristicas_faltantes:
            st.warning(f"Faltan {len(caracteristicas_faltantes)} características en el CSV.")
            if len(caracteristicas_faltantes) <= 5:
                st.warning(f"Características faltantes: {caracteristicas_faltantes}")
            else:
                st.warning(f"Primeras 5 faltantes: {list(caracteristicas_faltantes)[:5]}")

        # Crear DataFrame con características en el orden correcto
        datos_pred = {}
        for caract in caracteristicas_modelo:
            if caract in df.columns:
                datos_pred[caract] = df[caract].values
            else:
                # Si una característica falta, rellenar con ceros (o con la media/mediana adecuada)
                # OJO: Si muchas características faltan, esto puede afectar la predicción
                st.warning(f"La característica '{caract}' no se encontró en el CSV. Se rellenará con ceros.")
                datos_pred[caract] = np.zeros(len(df))

        # Crear DataFrame de una sola vez
        X_pred = pd.DataFrame(datos_pred)

        # Escalar características
        X_scaled = scaler.transform(X_pred)

    # Realizar predicciones
    with st.spinner("Realizando predicciones..."):
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

    # Determinar tipo de emoción basado en el nombre del archivo original (usado como proxy aquí)
    # Ya que los modelos están cargados en variables específicas (arousal_model_components, valence_model_components),
    # podemos usar ese conocimiento directamente.
    
    # Para saber si es arousal o valence, podemos pasar un parámetro adicional a la función
    # o inferirlo de alguna manera. Aquí lo haremos basándonos en cómo se llama la función.
    # Por ahora, la inferencia se hará al llamar a la función en la parte principal del script.
    
    return resultados


# Crear secciones para la aplicación
st.header("1. Cargar archivo CSV con características")

# Opción para cargar archivo
uploaded_file = st.file_uploader("Selecciona un archivo CSV con características", type=["csv"])

if uploaded_file is not None:
    # Mostrar proceso de carga
    with st.spinner('Cargando y procesando archivo...'):
        # Leer el archivo cargado
        df = pd.read_csv(uploaded_file, delimiter=';')
        
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
    
    # Sección para seleccionar modelos
    st.header("2. Modelos cargados")
    st.info(f"Los modelos de Arousal y Valence han sido cargados desde el repositorio: `{HF_REPO_ID}`")
    st.write(f"- **Modelo Arousal**: `{MODEL_AROUSAL_FILENAME}`")
    st.write(f"- **Modelo Valence**: `{MODEL_VALENCE_FILENAME}`")
    
    # Botón para iniciar predicciones
    if st.button("Realizar Predicciones"):
        st.header("3. Resultados de Predicciones")
        
        try:
            # Predicción de arousal
            with st.spinner("Realizando predicciones de Arousal..."):
                resultados_arousal = predecir_emocion_desde_csv(
                    df, 
                    arousal_model_components, # Pasamos los componentes cargados
                    columna_id
                )
            resultados_arousal.rename(columns={'prediccion': 'arousal'}, inplace=True)
            
            # Predicción de valence
            with st.spinner("Realizando predicciones de Valence..."):
                resultados_valence = predecir_emocion_desde_csv(
                    df, 
                    valence_model_components, # Pasamos los componentes cargados
                    columna_id
                )
            resultados_valence.rename(columns={'prediccion': 'valence'}, inplace=True)
            
            # Combinar resultados
            if columna_id:
                resultados_combinados = resultados_arousal.merge(
                    resultados_valence, on='ID'
                )
            else:
                # Si no hay ID, asumir mismo orden
                resultados_combinados = resultados_arousal.copy()
                resultados_combinados['valence'] = resultados_valence['valence']
            
            # Mostrar estadísticas
            st.subheader("Estadísticas de predicciones")
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("Media de Arousal", f"{resultados_combinados['arousal'].mean():.4f}")
                st.metric("Mínimo de Arousal", f"{resultados_combinados['arousal'].min():.4f}")
                st.metric("Máximo de Arousal", f"{resultados_combinados['arousal'].max():.4f}")
            
            with col_stats2:
                st.metric("Media de Valence", f"{resultados_combinados['valence'].mean():.4f}")
                st.metric("Mínimo de Valence", f"{resultados_combinados['valence'].min():.4f}")
                st.metric("Máximo de Valence", f"{resultados_combinados['valence'].max():.4f}")
            
            # Mostrar resultados en tabla
            st.subheader("Tabla de Predicciones")
            st.dataframe(resultados_combinados)
            
            # Visualización
            st.subheader("Visualización de Predicciones")
            
            # Crear visualización
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = sns.scatterplot(
                x='valence', 
                y='arousal', 
                data=resultados_combinados,
                alpha=0.7,
                ax=ax
            )
            
            # Añadir líneas de referencia
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            
            # Etiquetas
            plt.xlabel('Valence', fontsize=12)
            plt.ylabel('Arousal', fontsize=12)
            plt.title('Mapa Emocional: Arousal vs Valence', fontsize=14)
            
            # Añadir etiquetas a los cuadrantes
            plt.text(0.25, 0.75, 'Tenso', fontsize=12, ha='center')
            plt.text(0.75, 0.75, 'Alegre', fontsize=12, ha='center')
            plt.text(0.25, 0.25, 'Triste', fontsize=12, ha='center')
            plt.text(0.75, 0.25, 'Calmado', fontsize=12, ha='center')
            
            st.pyplot(fig)
            
            # Opción para descargar resultados
            csv = resultados_combinados.to_csv(index=False)
            st.download_button(
                label="Descargar predicciones como CSV",
                data=csv,
                file_name="predicciones_emocionales.csv",
                mime="text/csv",
            )
            
        except Exception as e:
            st.error(f"Error al realizar predicciones: {str(e)}")
            st.exception(e)

# Agregar información adicional
with st.sidebar:
    st.title("Información")
    st.markdown("""
    ### Sobre esta aplicación
    
    Esta herramienta permite predecir las dimensiones emocionales de **arousal** y **valence** en muestras musicales utilizando modelos de machine learning previamente entrenados.
    
    - **Arousal**: Nivel de activación o energía (alto vs. bajo)
    - **Valence**: Nivel de positividad o negatividad (positivo vs. negativo)
    
    ### Instrucciones
    
    1. Sube un archivo CSV con las características extraídas de tus muestras musicales
    2. Los modelos de predicción se cargarán automáticamente desde Hugging Face Hub.
    3. Haz clic en "Realizar Predicciones"
    4. Explora los resultados y descarga las predicciones
    
    ### Requisitos del CSV
    
    El CSV debe contener columnas con las características necesarias para los modelos. Opcionalmente, puede incluir una columna de identificación ('song_id', 'sample', 'ID', 'filename').
    """)
    
    st.markdown("---")
    st.markdown("Desarrollado para análisis de emociones musicales")