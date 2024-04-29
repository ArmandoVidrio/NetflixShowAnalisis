import streamlit as st

## Definimos configuraciones de nuestra pagina

st.set_page_config(
    page_title = "StreamRec"
)

st.title("StreamRec")
st.sidebar.success("Elige una plataforma para ver sus peliculas y series disponibles")

with st.expander("¿Qué es StreamRec?"):
    """
    # Aplicación de Recomendación de Series y Películas por Plataforma de Streaming

    ## Descripción
    La aplicación **StreamRec** es una herramienta interactiva diseñada para proporcionar recomendaciones personalizadas de series y películas basadas en análisis de datos de múltiples plataformas de streaming. Con **StreamRec**, los usuarios pueden descubrir nuevas series y películas que se adapten a sus preferencias y gustos en función de las plataformas que utilizan.

    ## Páginas
    1. **Netflix**: Esta página muestra recomendaciones de series y películas disponibles en Netflix, basadas en análisis de datos como género, calificaciones, popularidad y más.
    2. **Amazon Prime Video**: Aquí encontrarás recomendaciones exclusivas para series y películas de Amazon Prime Video, seleccionadas a partir de análisis detallados de diferentes métricas.
    3. **Disney+**: Descubre las mejores opciones en series y películas de Disney+ con recomendaciones generadas por algoritmos inteligentes que tienen en cuenta las preferencias del usuario y las tendencias de la plataforma.
    4. **Hulu**: Explora las últimas recomendaciones de series y películas de Hulu, con análisis profundos de contenido para ofrecer sugerencias precisas y relevantes.

    ## Características
    - **Interfaz Amigable**: Una interfaz intuitiva y fácil de usar que permite a los usuarios navegar entre las diferentes páginas de las plataformas de streaming.
    - **Resultados Personalizados**: Las recomendaciones se personalizan según las preferencias del usuario, como géneros favoritos, calificaciones anteriores, historial de visualización y más.
    - **Análisis Avanzado de Datos**: Utiliza análisis avanzados de DataFrames para evaluar el contenido y proporcionar recomendaciones precisas y relevantes.
    - **Filtrado y Ordenamiento**: Opciones de filtrado y ordenamiento para refinar las recomendaciones según criterios específicos como año de lanzamiento, duración, clasificación por edades, etc.
    - **Visualización de Datos**: Gráficos y visualizaciones que muestran estadísticas clave sobre las recomendaciones, como calificaciones promedio, popularidad, géneros más comunes, etc.

    ## Objetivo
    El objetivo principal de **StreamRec** es ofrecer a los usuarios una experiencia de descubrimiento de contenido más personalizada y eficiente, facilitando la búsqueda y selección de series y películas de alta calidad en las plataformas de streaming más populares.

    """