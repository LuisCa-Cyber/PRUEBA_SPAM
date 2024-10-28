import streamlit as st
import pickle
import joblib
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC


# Suponemos que el modelo y el vectorizador están preentrenados y cargados
# Puedes usar `pickle.load(open('modelo_entrenado.pkl', 'rb'))` para cargarlos
# Aquí inicializamos `CountVectorizer` y `SVC` solo como referencia

# Cargar el modelo y el vectorizador entrenado
# with open('C:/Users/Luis.fernandez/01. Python/13. Proyecto ML y NLP/modelo_entrenado_comprimido.pkl', 'rb') as model_file:
#     svm_model = pickle.load(model_file)

svm_model = joblib.load('modelo_entrenado_comprimido.pkl')

with open('vectorizador_entrenado.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

#cv = CountVectorizer()  # Sustituir por el CountVectorizer entrenado
#svm_model = SVC(probability=True)  # Sustituir por el modelo entrenado

# Instanciar LimeTextExplainer
explainer = LimeTextExplainer(class_names=['HAM', 'SPAM'])

# Configuración de la aplicación Streamlit
st.title("Detector de Spam con Explicación")
st.write("Ingrese un mensaje y presione el botón para verificar si es spam o no.")

# Entrada de texto del usuario
input_text = st.text_area("Escriba su mensaje aquí:")

# Función para realizar la predicción e interpretación
def interpretar_mensaje(mensaje):
    # Transformar el mensaje usando el vectorizador
    vector_mensaje = cv.transform([mensaje]).toarray()
    # Obtener la explicación de LIME
    exp = explainer.explain_instance(
        mensaje,
        lambda x: svm_model.predict_proba(cv.transform(x).toarray()),
        num_features=10
    )
    return exp

# Botón para ejecutar la verificación de spam
if st.button("Verificar si es Spam"):
    if input_text:
        # Ejecutar la interpretación
        explicacion = interpretar_mensaje(input_text)
        
        # Mostrar resultado de la predicción
        st.write("Explicación de la predicción:")
        st.components.v1.html(explicacion.as_html(), height=800)
    else:
        st.write("Por favor, ingrese un mensaje para analizar.")
