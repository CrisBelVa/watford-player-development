# Configuración de la página DEBE SER LO PRIMERO
import os
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Obtener la ruta absoluta al directorio del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, 'img')
LOGO_PATH = os.path.join(IMG_DIR, 'watford_logo.png')

# Configuración de la página - DEBE SER EL PRIMER COMANDO DE STREAMLIT
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

# Global CSS tweak requested
st.markdown(
    """
    <style>
      .st-emotion-cache-zy6yx3 { padding: 0rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar header logo via CSS background
def inject_sidebar_logo():
    try:
        img = Image.open(LOGO_PATH)
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(
            f"""
            <style>
            div[data-testid=\"stSidebarHeader\"] {{
                background-image: url('data:image/png;base64,{b64}');
                background-repeat: no-repeat;
                background-position: center;
                background-size: 90% auto;
                min-height: 100px;
                margin-bottom: 0.25rem;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass

inject_sidebar_logo()

# Main title
st.title("Watford Staff Dashboard")

# Navegación
page_options = ["Dashboard", "Individual Development"]

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

# Redirigir a individual_development si se selecciona esa opción
if page_option == "Individual Development":
    st.switch_page("pages/individual_development.py")

# Contenido principal del dashboard
st.write("This is the staff dashboard where you can manage player development, view analytics, and more.")
