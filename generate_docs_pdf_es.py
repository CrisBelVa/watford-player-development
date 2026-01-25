from fpdf import FPDF
import os

class TechnicalDocPDF(FPDF):
    def header(self):
        # Logo
        logo_path = '/Users/cristhiancamilobeltranvalencia/Documents/watford-player-development/img/watford_logo.png'
        if os.path.exists(logo_path):
            self.image(logo_path, 10, 8, 20)
        
        # Title
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Watford Player Development Hub', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()} | Documentación Técnica - Watford FC', 0, 0, 'C')

    def chapter_title(self, num, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(252, 236, 3) # Watford Yellow
        self.cell(0, 6, f'{num}. {label}', 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_table(self, header, data):
        self.set_font('Arial', 'B', 10)
        # Calculate column widths
        num_cols = len(header)
        col_width = (self.w - 20) / num_cols
        
        for col in header:
            self.cell(col_width, 7, col, 1)
        self.ln()
        
        self.set_font('Arial', '', 9)
        for row in data:
            for item in row:
                self.cell(col_width, 6, str(item), 1)
            self.ln()
        self.ln()

def generate_pdf():
    pdf = TechnicalDocPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Descripción General
    pdf.chapter_title(1, 'Descripción General del Proyecto')
    overview_text = (
        "El Watford Player Development Hub es una plataforma integral de análisis deportivo diseñada para rastrear, "
        "gestionar y optimizar el desarrollo individual de los jugadores dentro del Watford Football Club. Al integrar "
        "datos de rendimiento en partidos con actividades de entrenamiento individual, la plataforma proporciona una "
        "visión de 360 grados del progreso del jugador.\n\n"
        "Problemas Centrales Resueltos:\n"
        "- Centralización de datos de rendimiento y desarrollo.\n"
        "- Identificación de KPIs específicos por posición.\n"
        "- Seguimiento del desarrollo a largo plazo independiente de los cambios de personal."
    )
    pdf.chapter_body(overview_text)

    # 2. Stack Tecnológico
    pdf.chapter_title(2, 'Stack Tecnológico y Enfoque Técnico')
    tech_text = (
        "La plataforma está construida como una aplicación web modular basada en Python utilizando Streamlit. "
        "Se conecta a una base de datos MySQL centralizada (alojada en AWS RDS) y emplea una arquitectura de múltiples capas.\n\n"
        "Tecnologías Principales:\n"
        "- Python: Lenguaje principal para procesamiento de datos y lógica.\n"
        "- Streamlit: Marco de aplicación web.\n"
        "- MySQL (AWS RDS): Almacenamiento de base de datos relacional.\n"
        "- SQLAlchemy: Abstracción de base de datos y ORM.\n"
        "- Pandas / NumPy: Manipulación de datos y cómputo numérico.\n"
        "- Altair / Plotly / Matplotlib: Visualizaciones interactivas.\n"
        "- FPDF: Generación de PDF para informes."
    )
    pdf.chapter_body(tech_text)

    # 3. Fuentes de Datos
    pdf.chapter_title(3, 'Fuentes de Datos y Flujo de Datos')
    data_flow_text = (
        "La plataforma ingiere datos de sistemas de rendimiento de partidos (nivel de partido y evento) "
        "y registros de desarrollo (entrenamiento, reuniones, revisiones de video).\n\n"
        "Pasos del Flujo de Datos:\n"
        "1. Autenticación: Identifica el rol del usuario y el ID del jugador asociado.\n"
        "2. Ejecución de Consultas: Obtiene datos relevantes a través de SQLAlchemy.\n"
        "3. Normalización: Pandas maneja la limpieza y el mapeo de posiciones.\n"
        "4. Cálculo: La capa de lógica de negocio calcula KPIs y métricas avanzadas (xG, etc.).\n"
        "5. Renderizado: Se generan visualizaciones para la interfaz de usuario."
    )
    pdf.chapter_body(data_flow_text)

    # 4. Esquema de Base de Datos
    pdf.chapter_title(4, 'Tablas de la Base de Datos y Explicación del Esquema')
    schema_header = ['Nombre Tabla', 'Propósito', 'Columnas Clave']
    schema_data = [
        ['player_data', 'Info estática y apariciones', 'playerId, matchId'],
        ['player_stats', 'Estadísticas nivel partido', 'matchId, playerId'],
        ['event_data', 'Acciones granulares evento', 'matchId, x, y, typeId'],
        ['match_data', 'Metadatos de partidos', 'matchId, startDate'],
        ['jugadores', 'Lista maestra jugadores', 'id, nombre, activo'],
        ['meetings', 'Reuniones staff-jugador', 'jugador_id, fecha']
    ]
    pdf.add_table(schema_header, schema_data)

    # 5. Autenticación
    pdf.chapter_title(5, 'Autenticación y Lógica de Inicio de Sesión')
    auth_text = (
        "El sistema emplea un mecanismo de inicio de sesión personalizado:\n"
        "- Inicio de Sesión Staff: Autentica contra staff_users.csv usando usuario/contraseña.\n"
        "- Inicio de Sesión Jugador: Selección de nombre + verificación de contraseña maestra.\n"
        "- Manejo de Sesión: Utiliza st.session_state para la persistencia.\n"
        "- Seguridad: El control de acceso basado en roles garantiza la privacidad de los datos."
    )
    pdf.chapter_body(auth_text)

    # 6. Sección Staff
    pdf.chapter_title(6, 'Sección de Inicio de Sesión del Staff')
    staff_text = (
        "Propósito: Supervisión de alto nivel y gestión administrativa.\n\n"
        "Secciones Clave:\n"
        "- Panel General: Resumen de actividades totales y tasas de participación.\n"
        "- Desarrollo Individual: Historial detallado de actividad del jugador.\n"
        "- Gestionar Jugadores: Activación/desactivación y sincronización de base de datos.\n"
        "- Archivos: Importación y validación de Excel para registros de desarrollo."
    )
    pdf.chapter_body(staff_text)

    # 7. Sección Jugador
    pdf.chapter_title(7, 'Sección de Inicio de Sesión del Jugador')
    player_text = (
        "Propósito: Autoconocimiento y comparativa de rendimiento.\n\n"
        "Secciones Clave:\n"
        "- Estadísticas Resumen: KPIs específicos de posición con deltas.\n"
        "- Estadísticas Tendencias: Historia visual de rendimiento. Datos ordenados cronológicamente. "
        "Una línea roja discontinua representa el Promedio de la Temporada.\n"
        "- Comparación Jugadores: Comparativa con mejores jugadores. El sistema selecciona los 5 mejores equipos "
        "por puntos y extrae los 5 mejores jugadores en la misma posición por calificación total.\n"
        "- Detalles Jugador: Desglose completo partido a partido."
    )
    pdf.chapter_body(player_text)

    # 8. Desglose de Métricas
    pdf.chapter_title(8, 'Métricas - Desglose Técnico')
    metrics_text = (
        "Lógica de Cálculo:\n"
        "- Deltas: (Promedio Filtrado - Promedio Temporada).\n"
        "- Porcentajes Ponderados: Calculados a partir de conteos brutos.\n"
        "- Métricas Avanzadas: xG heurístico basado en coordenadas; Acciones progresivas basadas en distancia.\n\n"
        "Lógica de Tendencias y Comparación:\n"
        "- Línea Base Tendencias: Línea roja discontinua = (Suma de Valores / Total Partidos).\n"
        "- Selección Comparativa: Top 5 equipos por puntos -> Filtrar por posición -> Clasificar por total_rating."
    )
    pdf.chapter_body(metrics_text)

    # 9. Principios de Diseño
    pdf.chapter_title(9, 'Principios de Diseño y UX')
    design_text = (
        "- Priorización de KPIs: Las métricas se filtran por posición para reducir la carga cognitiva.\n"
        "- Consistencia: Lenguaje visual unificado en todos los paneles.\n"
        "- Accesibilidad: Gráficos y tablas claros para usuarios no técnicos."
    )
    pdf.chapter_body(design_text)

    # 10. Limitaciones
    pdf.chapter_title(10, 'Limitaciones y Mejoras Futuras')
    limit_text = (
        "Restricciones Conocidas:\n"
        "- Importaciones manuales de Excel para registros de desarrollo.\n"
        "- Dependencia de las tasas de actualización de datos de origen.\n\n"
        "Hoja de Ruta Futura:\n"
        "- Integración de API automatizada con proveedores de seguimiento.\n"
        "- Analítica predictiva para trayectorias de desarrollo.\n"
        "- Interfaz móvil para el registro de actividades."
    )
    pdf.chapter_body(limit_text)

    # 11. Desglose de KPIs por Posición
    pdf.chapter_title(11, 'Desglose de KPIs por Posición')
    
    # Tabla Porteros
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.1 Porteros', 0, 1, 'L')
    gk_header = ['KPI', 'Definición', 'Cálculo']
    gk_data = [
        ['totalSaves', 'Paradas Totales', 'Conteo de tiros detenidos'],
        ['save_pct', '% Paradas', '(Paradas / Tiros Puerta) * 100'],
        ['goals_conceded', 'Goles Concedidos', 'Total goles permitidos'],
        ['claimsHigh', 'Salidas por Alto', 'Centros altos atrapados'],
        ['collected', 'Balones Recogidos', 'Balones sueltos recogidos'],
        ['def_actions_outside_box', 'Acciones Líbero', 'Acciones fuera del área'],
        ['ps_xG', 'xG Post-Tiro', 'Evalúa colocación del tiro']
    ]
    pdf.add_table(gk_header, gk_data)

    # Tabla Defensas
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.2 Defensas (Laterales y Centrales)', 0, 1, 'L')
    df_header = ['KPI', 'Definición', 'Cálculo']
    df_data = [
        ['interceptions', 'Intercepciones', 'Pases oponentes cortados'],
        ['progressive_passes', 'Pases Prog.', '10+ yardas hacia portería'],
        ['recoveries', 'Recuperaciones', 'Ganar posesión balón suelto'],
        ['crosses', 'Centros', 'Balones al área de penalti'],
        ['take_on_success_pct', '% Éxito Regate', '(Exitosos / Total) * 100'],
        ['pass_completion_pct', '% Precisión Pases', '(Exitosos / Total) * 100'],
        ['clearances', 'Despejes', 'Alejar balón del peligro'],
        ['long_pass_pct', '% Pases Largos', '(Exitosos / Total) * 100'],
        ['aerial_duel_pct', '% Duelos Aéreos', '(Ganados / Total) * 100']
    ]
    pdf.add_table(df_header, df_data)

    # Tabla Mediocampistas
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.3 Mediocampistas', 0, 1, 'L')
    mf_header = ['KPI', 'Definición', 'Cálculo']
    mf_data = [
        ['recoveries', 'Recuperaciones', 'Ganar posesión balón suelto'],
        ['interceptions', 'Intercepciones', 'Pases oponentes cortados'],
        ['pass_completion_pct', '% Precisión Pases', '(Exitosos / Total) * 100'],
        ['progressive_passes', 'Pases Prog.', '10+ yardas hacia portería'],
        ['key_passes', 'Pases Clave', 'Pases que conducen a tiro'],
        ['passes_into_penalty_area', 'Pases al Área', 'Pases al área oponente'],
        ['goal_creating_actions', 'Creación Goles', 'Acciones que llevan a gol'],
        ['shot_creating_actions', 'Creación Tiros', 'Acciones que llevan a tiro']
    ]
    pdf.add_table(mf_header, mf_data)

    # Tabla Delanteros
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.4 Delanteros (Extremos y Atacantes)', 0, 1, 'L')
    fw_header = ['KPI', 'Definición', 'Cálculo']
    fw_data = [
        ['goals', 'Goles', 'Número de goles marcados'],
        ['assists', 'Asistencias', 'Número de asistencias'],
        ['xG', 'Goles Esperados', 'Prob. de que tiro sea gol'],
        ['xA', 'Asistencias Esp.', 'Prob. de que pase sea gol'],
        ['shots_on_target_pct', '% Precisión Tiros', '(Puerta / Total) * 100'],
        ['carries_into_box', 'Conducciones Área', 'Llevar balón al área'],
        ['take_on_success_pct', '% Éxito Regate', '(Exitosos / Total) * 100'],
        ['goal_creating_actions', 'Creación Goles', 'Acciones que llevan a gol']
    ]
    pdf.add_table(fw_header, fw_data)

    output_path = '/Users/cristhiancamilobeltranvalencia/Documents/watford-player-development/Documentacion_Tecnica_Proyecto.pdf'
    pdf.output(output_path)
    print(f"PDF generado con éxito en: {output_path}")

if __name__ == "__main__":
    generate_pdf()
