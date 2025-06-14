import streamlit as st
from PIL import Image

# --- Page settings ---
import os

# Obtener la ruta absoluta al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')

st.set_page_config(
    page_title="Watford Staff Dashboard",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"  # Mostrar sidebar para usuarios logueados
)

# Verificar autenticación
if "logged_in" not in st.session_state or not st.session_state.logged_in or st.session_state.user_type != "staff":
    st.warning("You must be logged in as staff to view this page.")
    st.stop()

# Mostrar logo y título
try:
    logo = Image.open(LOGO_PATH)
except FileNotFoundError:
    st.error("Logo image not found. Please check the image path.")
    logo = None
st.image(logo, width=100)
st.title("Watford Staff Dashboard")

# Navegación
page_options = ["🏠 Dashboard", "🎯 Individual Development"]

# Mostrar información del usuario en el sidebar
with st.sidebar:
    st.subheader("Navigation")
    page_option = st.radio(
        "Go to",
        page_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.subheader("Staff Info")
    if "staff_info" in st.session_state:
        st.write(f"Name: {st.session_state.staff_info['full_name']}")
        st.write(f"Role: {st.session_state.staff_info['role']}")
        
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Cargar la página seleccionada
if page_option == "🏠 Dashboard":
    # Contenido principal del dashboard
    st.header("Welcome to Staff Dashboard")
    st.write("This is the staff dashboard where you can manage player development, view analytics, and more.")
    
elif page_option == "🎯 Individual Development":
    # Cargar el módulo de desarrollo individual
    import sys
    import os
    
    # Asegurarse de que el directorio pages esté en el path
    pages_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if pages_dir not in sys.path:
        sys.path.append(pages_dir)
    
    # Importar y ejecutar el módulo
    try:
        import individual_development
    except ImportError as e:
        st.error(f"Error al cargar el módulo de desarrollo individual: {e}")
        st.error(f"Directorio actual: {os.getcwd()}")
        st.error(f"Contenido del directorio: {os.listdir(pages_dir)}")
