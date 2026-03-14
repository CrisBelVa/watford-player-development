from fpdf import FPDF
import pandas as pd
from datetime import datetime
import os
import tempfile
import plotly.io as pio
import numpy as np  # ← AÑADIR ESTA LÍNEA
import plotly.graph_objects as go  # ← AÑADIR ESTA LÍNEA

class WatfordPlayerReport(FPDF):
    """
    Generador de reportes PDF para jugadores de Watford FC
    """
    
    def __init__(self, player_name, logo_path, background_image_path=None):  # ← NUEVO PARÁMETRO
        super().__init__(orientation='L', unit='mm', format='A4')  # Landscape
        self.player_name = player_name
        self.logo_path = logo_path
        self.background_image_path = background_image_path  # ← NUEVO
        self.set_auto_page_break(auto=True, margin=15)
        
        # Colores Watford
        self.COLOR_YELLOW = (252, 236, 3)
        self.COLOR_PINK = (255, 230, 230)
        self.COLOR_GRAY_BG = (50, 50, 50)  # ← NUEVO
        self.COLOR_GRAY = (136, 136, 136)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_RED = (237, 28, 36)  # ← NUEVO

        
    def header(self):
        """Header con fondo gris y nombre blanco (estilo Individual Development)"""
        if self.page_no() > 1:  # No header en portada
            # Fondo gris
            self.set_fill_color(*self.COLOR_GRAY_BG)
            self.rect(0, 0, 297, 210, 'F')  # ← Landscape: 297mm ancho
            
            # Nombre del jugador (centrado verticalmente a la derecha)
            self.set_xy(150, 12)
            self.set_font('Arial', 'B', 26)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(110, 15, self.player_name.upper(), 0, 0, 'R')
            
            # Logo a la derecha
            try:
                if os.path.exists(self.logo_path):
                    self.image(self.logo_path, x=270, y=8, w=20)
            except Exception as e:
                print(f"Error loading logo in header: {e}")
            
            self.set_y(35)
    
    def footer(self):
        """Footer con texto blanco sobre fondo gris"""
        if self.page_no() > 1:  # No footer en portada
            self.set_y(-15)
            
            # "Player Stats Report" centrado (texto blanco)
            self.set_font('Arial', 'I', 10)
            self.set_text_color(*self.COLOR_WHITE)  # ← CAMBIO
            self.cell(0, 10, 'Player Stats Report', 0, 0, 'C')
            
            # Número de página a la derecha
            self.set_font('Arial', 'I', 10)
            page_text = f'Page {self.page_no() - 1}'
            self.cell(0, 10, page_text, 0, 0, 'R')

    def cover_page(self):
        """
        Portada estilo MAX ALLEYNE (Individual Development)
        - Franja gris izquierda con logo
        - Línea roja vertical
        - Imagen de fondo (Watford_portada_d.jpg)
        - Nombre del jugador en blanco
        """
        self.add_page()
        
        # Franja gris izquierda
        self.set_fill_color(*self.COLOR_GRAY_BG)
        self.rect(0, 0, 60, 210, 'F')
        
        # Línea roja vertical
        self.set_fill_color(*self.COLOR_RED)
        self.rect(60, 0, 3, 210, 'F')
        
        # Logo en franja gris
        try:
            if os.path.exists(self.logo_path):
                self.image(self.logo_path, x=10, y=15, w=40)
        except:
            pass
        
        # Texto "PLAYER STATS\nREPORT" en franja gris
        self.set_xy(5, 90)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_WHITE)
        self.multi_cell(50, 8, 'PLAYER STATS\nREPORT', 0, 'C')
        
        # Nombre del autor/generador
        self.set_xy(5, 180)
        self.set_font('Arial', '', 12)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(50, 8, 'ALBERTO GARRIDO', 0, 0, 'C')
        
        # Imagen de fondo (Watford_portada_d.jpg)
        try:
            if self.background_image_path and os.path.exists(self.background_image_path):
                # ✅ NO forzar ancho completo - dejar que se centre naturalmente
                # La imagen tiene el escudo centrado, así que usamos su proporción original
                self.image(self.background_image_path, x=63, y=0, h=210)  # ← SOLO altura, NO ancho
        except Exception as e:
            print(f"Error loading background: {e}")
        
        # Nombre del jugador (sobre la imagen, arriba a la derecha)
        self.set_xy(150, 20)
        self.set_font('Arial', 'B', 32)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(130, 15, self.player_name.upper(), 0, 0, 'R')
    
    def filters_page(self, filters_data):
        """
        Página 2: Filtros aplicados (LETRAS BLANCAS)
        - Season
        - Date Range
        - Matches seleccionados
        """
        self.add_page()
        
        # Título (BLANCO)
        self.set_font('Arial', 'B', 20)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(0, 15, 'Filters Applied', 0, 1, 'L')
        self.ln(5)
        
        # Season (BLANCO)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(60, 10, 'Season:', 0, 0, 'L')
        self.set_font('Arial', '', 14)
        season = filters_data.get('season', 'All seasons')
        if season is None or str(season).lower() == 'none':
            season = 'All seasons'
        self.cell(0, 10, str(season), 0, 1, 'L')
        self.ln(3)
        
        # Main Range (BLANCO)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(60, 10, 'Main Range:', 0, 0, 'L')
        self.set_font('Arial', '', 14)
        delta_start = filters_data.get('delta_start_date', filters_data.get('start_date', 'N/A'))
        delta_end = filters_data.get('delta_end_date', filters_data.get('end_date', 'N/A'))
        if isinstance(delta_start, pd.Timestamp):
            delta_start = delta_start.strftime('%Y-%m-%d')
        if isinstance(delta_end, pd.Timestamp):
            delta_end = delta_end.strftime('%Y-%m-%d')
        self.cell(0, 10, f'{delta_start} to {delta_end}', 0, 1, 'L')
        self.ln(3)

        # Comparison Mode (BLANCO)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(60, 10, 'Comparison:', 0, 0, 'L')
        self.set_font('Arial', '', 14)
        comparison_mode = filters_data.get('comparison_mode', 'Previous month')
        comparison_value = filters_data.get('comparison_value', '')
        comparison_text = str(comparison_mode)
        if comparison_value:
            comparison_text = f"{comparison_text} ({comparison_value})"
        self.cell(0, 10, comparison_text, 0, 1, 'L')
        self.ln(3)

        # Reference Range (BLANCO)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(60, 10, 'Reference Range:', 0, 0, 'L')
        self.set_font('Arial', '', 14)
        ref_start = filters_data.get('reference_start_date', filters_data.get('start_date', 'N/A'))
        ref_end = filters_data.get('reference_end_date', filters_data.get('end_date', 'N/A'))
        if isinstance(ref_start, pd.Timestamp):
            ref_start = ref_start.strftime('%Y-%m-%d')
        if isinstance(ref_end, pd.Timestamp):
            ref_end = ref_end.strftime('%Y-%m-%d')
        self.cell(0, 10, f'{ref_start} to {ref_end}', 0, 1, 'L')
        self.ln(8)
        
        # Selected Matches (BLANCO)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(0, 10, 'Selected Matches:', 0, 1, 'L')
        self.ln(3)
        
        matches = filters_data.get('selected_matches', [])
        if matches:
            self.set_font('Arial', '', 11)
            self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
            
            # Crear tabla de matches (2 columnas)
            col_width = 135
            row_height = 8
            max_matches_to_show = 16  # keep filters page compact and avoid near-empty overflow page
            
            for i, match in enumerate(matches):
                # Limitar a primeros partidos y resumir el resto
                if i >= max_matches_to_show:
                    self.set_font('Arial', 'I', 10)
                    self.cell(0, 8, f'... and {len(matches) - max_matches_to_show} more matches', 0, 1, 'L')
                    break
                
                # Alternar columnas
                if i % 2 == 0:
                    x_pos = self.get_x()
                    y_pos = self.get_y()
                
                # Dibujar celda con borde (FONDO GRIS OSCURO)
                self.set_fill_color(60, 60, 60)  # ← GRIS OSCURO
                self.cell(col_width, row_height, f'  {match}', 1, 0, 'L', True)
                
                if i % 2 == 0:
                    # Primera columna
                    self.set_xy(x_pos + col_width + 2, y_pos)
                else:
                    # Segunda columna
                    self.ln()
            
            # Si terminó en columna 1, saltar línea
            if len(matches) % 2 == 1:
                self.ln()
        else:
            self.set_font('Arial', 'I', 12)
            self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
            self.cell(0, 10, 'No matches selected', 0, 1, 'L')
    
    def draw_info_card(self, x, y, width, height, label, value):
        """Dibuja una card de información personal (rosa)"""
        # Fondo rosa
        self.set_fill_color(*self.COLOR_PINK)
        self.rect(x, y, width, height, 'F')
        
        # Label
        self.set_xy(x, y + 3)
        self.set_font('Arial', '', 9)
        self.set_text_color(*self.COLOR_GRAY)
        self.cell(width, 5, label, 0, 0, 'C')
        
        # Value
        self.set_xy(x, y + 9)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(width, 5, str(value), 0, 0, 'C')
    
    def draw_metric_card(self, x, y, width, height, title, value, delta, delta_pct):
        """Dibuja una card de métrica KPI (amarilla) con delta"""
        # Borde amarillo
        self.set_draw_color(*self.COLOR_YELLOW)
        self.set_line_width(0.5)
        self.rect(x, y, width, height)
        
        # Título
        self.set_xy(x, y + 3)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(width, 5, title, 0, 0, 'C')
        
        # Valor principal
        self.set_xy(x, y + 10)
        self.set_font('Arial', '', 16)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(width, 7, str(value), 0, 0, 'C')
        
        # Delta
        self.set_xy(x, y + 19)
        self.set_font('Arial', '', 8)
        self.set_text_color(*self.COLOR_GRAY)
        
        # Flecha
        if delta > 0:
            arrow = "^"
        elif delta < 0:
            arrow = "v"
        else:
            arrow = ''
        
        delta_text = f'{arrow} {delta:+.1f} ({delta_pct:+.1f}%)'
        self.cell(width, 4, delta_text, 0, 0, 'C')
    
    def metrics_page(self, player_info, player_position, kpis_data):
        """
        Página 3: Info del jugador + Métricas KPI (TODAS EN UNA PÁGINA)
        - 7 cards de info personal (rosa)
        - Cards de métricas por posición (LETRAS BLANCAS, fondo gris)
        """
        self.add_page()
        
        # Nombre del jugador como título (BLANCO)
        self.set_font('Arial', 'B', 18)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(0, 12, self.player_name, 0, 1, 'L')
        self.ln(3)
        
        # Cards de información personal (7 cards en una fila)
        card_width = 38
        card_height = 18
        spacing = 2
        start_x = 15
        start_y = self.get_y()
        
        info_labels = ['Age', 'Shirt Number', 'Height', 'Weight', 'Games Played', 
                    'Games as Starter', 'Minutes Played']
        info_keys = ['age', 'shirtNo', 'height', 'weight', 'gamesPlayed', 
                    'gamesStarter', 'minutesPlayed']  # ← CORREGIDO NOMBRES
        
        for i, (label, key) in enumerate(zip(info_labels, info_keys)):
            x = start_x + (card_width + spacing) * i
            value = player_info.get(key, 'N/A')
            if key == 'minutesPlayed':
                value = int(value) if value != 'N/A' else 'N/A'
            self.draw_info_card(x, start_y, card_width, card_height, label, value)
        
        self.set_y(start_y + card_height + 8)  # ← Reducido espacio
        
        # "Showing Metrics for position: X" (BLANCO)
        self.set_font('Arial', 'I', 11)
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(0, 6, f'Showing Metrics for position: {player_position}', 0, 1, 'L')
        self.ln(1)
        
        # ===== CARDS DE MÉTRICAS KPI (5 POR FILA, LETRAS BLANCAS) =====
        kpi_card_width = 54  # ← MÁS PEQUEÑO (5 por fila)
        kpi_card_height = 22  # ← REDUCIDO
        kpi_spacing = 2
        kpis_per_row = 5  # ← 5 POR FILA (antes 4)
        
        kpi_start_x = 15
        kpi_start_y = self.get_y()
        
        for i, kpi_item in enumerate(kpis_data):
            row = i // kpis_per_row
            col = i % kpis_per_row
            
            x = kpi_start_x + (kpi_card_width + kpi_spacing) * col
            y = kpi_start_y + (kpi_card_height + kpi_spacing) * row
            
            self.draw_metric_card_white_text(  # ← NUEVA FUNCIÓN
                x, y, kpi_card_width, kpi_card_height,
                kpi_item['title'],
                kpi_item['value'],
                kpi_item['delta'],
                kpi_item['delta_pct']
        )
            
    def draw_metric_card_white_text(self, x, y, width, height, title, value, delta, delta_pct):
        """Dibuja una card de métrica KPI con LETRAS BLANCAS"""
        # Borde amarillo
        self.set_draw_color(*self.COLOR_YELLOW)
        self.set_line_width(0.5)
        self.rect(x, y, width, height)
        
        # Título (BLANCO)
        self.set_xy(x, y + 2)
        self.set_font('Arial', 'B', 8)  # ← Reducido
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        # Truncar título si es muy largo
        title_display = title if len(title) <= 20 else title[:18] + '...'
        self.cell(width, 4, title_display, 0, 0, 'C')
        
        # Valor principal (BLANCO)
        self.set_xy(x, y + 8)
        self.set_font('Arial', 'B', 14)  # ← Reducido
        self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
        self.cell(width, 6, str(value), 0, 0, 'C')
        
        # Delta (COLOR según signo)
        self.set_xy(x, y + 16)
        self.set_font('Arial', '', 7)  # ← Reducido
        
        # Color según delta
        if delta > 0:
            self.set_text_color(0, 200, 0)  # Verde brillante
            arrow = "^"
        elif delta < 0:
            self.set_text_color(255, 80, 80)  # Rojo brillante
            arrow = "v"
        else:
            self.set_text_color(*self.COLOR_WHITE)
            arrow = ''
        
        delta_text = f'{arrow} {delta:+.1f} ({delta_pct:+.1f}%)'
        self.cell(width, 3, delta_text, 0, 0, 'C')
    
    def stats_table_page(self, df, position_kpis):
        """
        Página 4: Tabla de estadísticas (LETRAS BLANCAS)
        Solo columnas relevantes para la posición
        """
        self.add_page()

        if df.empty:
            self.set_font('Arial', 'B', 18)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(0, 12, 'Player Stats', 0, 1, 'L')
            self.ln(3)
            self.set_font('Arial', 'I', 12)
            self.set_text_color(*self.COLOR_WHITE)  # ← BLANCO
            self.cell(0, 10, 'No data available', 0, 1, 'C')
            return
        
        # Columnas a mostrar: fecha, oponente + KPIs de posición
        base_cols = ['matchDate', 'oppositionTeamName']
        
        # Filtrar solo columnas que existen en el df
        available_kpis = [col for col in position_kpis if col in df.columns]
        table_columns = base_cols + available_kpis
        
        # Filtrar df
        df_table = df[table_columns].copy()

        # Some feeds store xA in centi-units (e.g., 15 == 0.15). Normalize only when detected.
        if "xA" in df_table.columns:
            xa_numeric = pd.to_numeric(df_table["xA"], errors="coerce")
            if xa_numeric.dropna().quantile(0.75) > 3:
                df_table["xA"] = xa_numeric / 100.0

        # Formatear fecha
        if 'matchDate' in df_table.columns:
            df_table['matchDate'] = pd.to_datetime(df_table['matchDate']).dt.strftime('%Y-%m-%d')
        
        # Formatear métricas numéricas (especialmente xG, xA)
        numeric_cols = ['xG', 'xA', 'ps_xG', 'progressive_passes', 'progressive_carry_distance']
        for col in numeric_cols:
            if col in df_table.columns:
                df_table[col] = pd.to_numeric(df_table[col], errors='coerce')
                df_table[col] = df_table[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "0.00")
        
        # Formatear porcentajes
        pct_cols = [c for c in df_table.columns if 'pct' in c or '%' in c]
        for col in pct_cols:
            if col in df_table.columns:
                df_table[col] = pd.to_numeric(df_table[col], errors='coerce')
                df_table[col] = df_table[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "0.0")
        
        # Formatear enteros
        int_cols = ['goals', 'assists', 'key_passes', 'passes_into_penalty_area', 
                   'carries_into_final_third', 'carries_into_penalty_area',
                   'goal_creating_actions', 'shot_creating_actions']
        for col in int_cols:
            if col in df_table.columns:
                df_table[col] = pd.to_numeric(df_table[col], errors='coerce')
                df_table[col] = df_table[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "0")
        
        # Renombrar columnas para headers más cortos
        column_labels = {
            'matchDate': 'Date',
            'oppositionTeamName': 'Opponent',
        }
        
        # Labels de métricas
        metric_labels = {
            "pass_completion_pct": "Pass%",
            "key_passes": "Key Pass",
            "aerial_duel_pct": "Aerial%",
            "take_on_success_pct": "Dribble%",
            "goal_creating_actions": "GCA",
            "shot_creating_actions": "SCA",
            "shots_on_target_pct": "Shot%",
            "passes_into_penalty_area": "Pass PA",
            "carries_into_final_third": "Carry F3",
            "carries_into_penalty_area": "Carry PA",
            "goals": "Goals",
            "assists": "Assists",
            "xG": "xG",
            "xA": "A.Shots",
        }
        
        for col in available_kpis:
            if col in metric_labels:
                column_labels[col] = metric_labels[col]
        
        df_table = df_table.rename(columns=column_labels)
        
        # Configurar tabla
        self.set_font('Arial', '', 7)
        
        # Calcular anchos de columna (ajustados a landscape)
        total_width = 277
        date_width = 25
        opponent_width = 40
        remaining_width = total_width - date_width - opponent_width
        metric_width = remaining_width / len(available_kpis) if available_kpis else 20
        
        col_widths = [date_width, opponent_width] + [metric_width] * len(available_kpis)

        # Section pagination label for stats table pages (e.g. 1/3, 2/3, 3/3).
        title_height = 12
        title_spacing = 3
        header_height = 8
        row_height = 6
        page_break_y = 175
        title_start_y = self.get_y()
        table_rows_start_y = title_start_y + title_height + title_spacing + header_height
        rows_per_page = max(1, int((page_break_y - table_rows_start_y) // row_height) + 1)
        total_table_pages = max(1, int(np.ceil(len(df_table) / rows_per_page)))
        current_table_page = 1

        def render_stats_table_header(page_index: int):
            suffix = f" {page_index}/{total_table_pages}" if total_table_pages > 1 else ""
            self.set_font('Arial', 'B', 18)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(0, title_height, f'Player Stats{suffix}', 0, 1, 'L')
            self.ln(title_spacing)

            self.set_fill_color(*self.COLOR_YELLOW)
            self.set_text_color(*self.COLOR_BLACK)
            self.set_font('Arial', 'B', 7)
            for col_name, width in zip(df_table.columns, col_widths):
                self.cell(width, header_height, str(col_name), 1, 0, 'C', True)
            self.ln()

            self.set_font('Arial', '', 6)
            self.set_text_color(*self.COLOR_WHITE)

        render_stats_table_header(current_table_page)
        
        for idx, row in df_table.iterrows():
            # Alternar color de fondo
            if idx % 2 == 0:
                self.set_fill_color(60, 60, 60)  # Gris oscuro
            else:
                self.set_fill_color(45, 45, 45)  # Gris muy oscuro
            
            for col_name, width in zip(df_table.columns, col_widths):
                value = row[col_name]
                # Truncar texto largo
                if isinstance(value, str) and len(value) > 15:
                    value = value[:12] + '...'
                self.cell(width, 6, str(value), 1, 0, 'C', True)  # ← SIEMPRE con fill
            self.ln()
            
            # Pagination: nueva página si nos quedamos sin espacio
            if self.get_y() > 175:
                self.add_page()
                current_table_page += 1
                render_stats_table_header(current_table_page)


    def trends_stats_page(self, trends_data):
        """
        Páginas de Trends Stats: 2 gráficos por página (AJUSTADO AL ESPACIO)
        """
        if not trends_data:
            return

        charts_per_page = 2
        charts_on_current_page = 0
        max_content_y = 188  # Reserve bottom area so footer/page number is never covered.
        desired_img_width = 270
        desired_img_height = 58

        def start_trends_page():
            self.add_page()
            self.set_font('Arial', 'B', 16)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(0, 10, 'Trends Stats', 0, 1, 'L')
            self.ln(2)

        start_trends_page()

        for item in trends_data:
            # Hard cap per page + dynamic cap by available vertical space.
            estimated_block_height = 6 + 1 + desired_img_height + 5
            if charts_on_current_page >= charts_per_page or (self.get_y() + estimated_block_height) > max_content_y:
                start_trends_page()
                charts_on_current_page = 0

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name

                pio.write_image(item['fig'], tmp_path, width=1400, height=500, scale=2)

                self.set_font('Arial', 'B', 10)
                self.set_text_color(*self.COLOR_WHITE)
                self.cell(0, 6, item['kpi_name'], 0, 1, 'L')
                self.ln(1)

                y_top = self.get_y()
                available_height = max_content_y - y_top
                img_height = min(desired_img_height, available_height)

                # If there is not enough room for a readable chart, move it to next page.
                if img_height < 35:
                    start_trends_page()
                    charts_on_current_page = 0
                    self.set_font('Arial', 'B', 10)
                    self.set_text_color(*self.COLOR_WHITE)
                    self.cell(0, 6, item['kpi_name'], 0, 1, 'L')
                    self.ln(1)
                    y_top = self.get_y()
                    img_height = min(desired_img_height, max_content_y - y_top)

                self.image(tmp_path, x=10, y=y_top, w=desired_img_width, h=img_height)
                self.ln(img_height + 5)
                charts_on_current_page += 1

            except Exception as e:
                print(f"Error adding trend chart: {e}")
                self.set_font('Arial', 'I', 10)
                self.set_text_color(*self.COLOR_WHITE)
                self.cell(0, 8, f"[Chart: {item['kpi_name']} - Could not render]", 0, 1, 'L')
                self.ln(5)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass


    def player_comparison_page(self, comparison_data, comparison_kpi_table, comparison_charts):
        """
        Páginas de Player Comparison
        
        Args:
            comparison_data: DataFrame con datos generales (Players Summary)
            comparison_kpi_table: DataFrame con KPIs comparados (playerId, playerName, teamName + KPIs)
            comparison_charts: Lista de dicts con gráficos comparativos
        """
        self.add_page()
        
        # Título de la sección
        self.set_font('Arial', 'B', 18)
        self.set_text_color(*self.COLOR_WHITE)
        self.cell(0, 12, 'Player Comparison', 0, 1, 'L')
        self.ln(3)
        
        # Subtítulo
        self.set_font('Arial', 'I', 12)
        self.set_text_color(*self.COLOR_GRAY)
        self.cell(0, 8, 'Top Players in Competition', 0, 1, 'L')
        self.ln(5)
        
        # ========== TABLA 1: PLAYERS SUMMARY ==========
        if not comparison_data.empty:
            self.set_font('Arial', 'B', 11)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 8, 'Players Summary', 0, 1, 'L')
            self.ln(2)
            
            # Headers de tabla
            col_widths = [60, 50, 20, 30, 30, 35]
            headers = ['Player', 'Team', 'Age', 'Games', 'Starter', 'Minutes']
            
            self.set_fill_color(*self.COLOR_YELLOW)
            self.set_font('Arial', 'B', 8)
            for header, width in zip(headers, col_widths):
                self.cell(width, 7, header, 1, 0, 'C', True)
            self.ln()
            
            # Datos de tabla
            self.set_font('Arial', '', 7)
            display_cols = ['Player', 'Team', 'Age', 'Games Played', 'Games as Starter', 'Minutes Played']
            
            for idx, row in comparison_data.head(10).iterrows():
                fill = idx % 2 == 0
                if fill:
                    self.set_fill_color(245, 245, 245)
                
                values = [
                    str(row.get('Player', ''))[:20],  # Truncar
                    str(row.get('Team', ''))[:15],
                    str(row.get('Age', '')),
                    str(row.get('Games Played', '')),
                    str(row.get('Games as Starter', '')),
                    str(int(row.get('Minutes Played', 0)))
                ]
                
                for value, width in zip(values, col_widths):
                    self.cell(width, 6, value, 1, 0, 'C', fill)
                self.ln()
            
            self.ln(8)
        
        # ========== TABLA 2: PLAYERS STATS KPI COMPARISON ==========
        if comparison_kpi_table is not None and not comparison_kpi_table.empty:
            self.set_font('Arial', 'B', 11)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 8, 'Players Stats KPI Comparison', 0, 1, 'L')
            self.ln(2)
            
            # Seleccionar columnas para mostrar
            base_cols = ['playerName', 'teamName']
            kpi_cols = [col for col in comparison_kpi_table.columns 
                       if col not in ['playerId', 'playerName', 'teamName']]
            
            # Limitar a máximo 8 KPIs para que quepa en la página
            kpi_cols = kpi_cols[:8]
            display_cols = base_cols + kpi_cols
            
            # Calcular anchos de columna dinámicamente
            total_width = 277
            name_width = 50
            team_width = 45
            remaining = total_width - name_width - team_width
            kpi_width = remaining / len(kpi_cols) if kpi_cols else 20
            
            col_widths = [name_width, team_width] + [kpi_width] * len(kpi_cols)
            
            # Labels cortos para headers
            short_labels = {
                'playerName': 'Player',
                'teamName': 'Team',
                'pass_completion_pct': 'Pass%',
                'key_passes': 'KeyP',
                'aerial_duel_pct': 'Aer%',
                'take_on_success_pct': 'Drib%',
                'goal_creating_actions': 'GCA',
                'shot_creating_actions': 'SCA',
                'shots_on_target_pct': 'Shot%',
                'passes_into_penalty_area': 'PassPA',
                'carries_into_final_third': 'CarF3',
                'carries_into_penalty_area': 'CarPA',
                'goals': 'G',
                'assists': 'A',
                'xG': 'xG',
                'xA': 'AShot',
            }
            
            # Headers
            self.set_fill_color(*self.COLOR_YELLOW)
            self.set_font('Arial', 'B', 7)
            for col, width in zip(display_cols, col_widths):
                label = short_labels.get(col, col[:6])
                self.cell(width, 7, label, 1, 0, 'C', True)
            self.ln()
            
            # Datos
            self.set_font('Arial', '', 6)
            for idx, row in comparison_kpi_table.head(10).iterrows():
                fill = idx % 2 == 0
                if fill:
                    self.set_fill_color(245, 245, 245)
                
                for col, width in zip(display_cols, col_widths):
                    value = row.get(col, '')
                    
                    # Formatear valores
                    if col in base_cols:
                        # Nombres: truncar
                        value_str = str(value)[:15] if col == 'playerName' else str(value)[:12]
                    elif isinstance(value, (int, float)):
                        # Números: formatear según tipo
                        if 'pct' in col or '%' in col:
                            value_str = f"{value:.1f}"
                        elif col in ['xG', 'xA']:
                            value_str = f"{value:.2f}"
                        else:
                            value_str = f"{int(value)}"
                    else:
                        value_str = str(value)
                    
                    self.cell(width, 6, value_str, 1, 0, 'C', fill)
                self.ln()
            
            self.ln(5)
        
        # ========== GRÁFICOS COMPARATIVOS (3 POR PÁGINA) ==========
        if comparison_charts:
            charts_per_page = 3  # ← CAMBIO
            chart_count = 0
            
            for item in comparison_charts:
                # Nueva página si ya tenemos 3 gráficos (y no es el primero)
                if chart_count > 0 and chart_count % charts_per_page == 0:
                    self.add_page()
                    
                    # Re-imprimir título
                    self.set_font('Arial', 'B', 18)
                    self.set_text_color(*self.COLOR_WHITE)
                    self.cell(0, 12, 'Player Comparison', 0, 1, 'L')
                    self.ln(3)
                
                try:
                    # Guardar gráfico como imagen temporal
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Convertir Plotly a imagen
                    pio.write_image(item['fig'], tmp_path, width=1200, height=400, scale=2)
                    
                    # Título del gráfico
                    self.set_font('Arial', 'B', 10)
                    self.set_text_color(*self.COLOR_WHITE)
                    self.cell(0, 6, item['kpi_name'], 0, 1, 'L')
                    self.ln(1)
                    
                    # Insertar imagen (ajustado para 3)
                    img_width = 270
                    img_height = 58
                    
                    self.image(tmp_path, x=10, y=self.get_y(), w=img_width, h=img_height)
                    self.ln(img_height + 4)
                    
                    # Eliminar archivo temporal
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
                    chart_count += 1
                    
                except Exception as e:
                    print(f"Error adding comparison chart: {e}")
                    self.set_font('Arial', 'I', 10)
                    self.set_text_color(*self.COLOR_WHITE)
                    self.cell(0, 8, f"[Chart: {item['kpi_name']} - Could not render]", 0, 1, 'L')
                    self.ln(5)


def generate_player_report(
    player_name,
    player_info,
    player_position,
    aggregated_metrics,
    filtered_df,
    filters_data,
    logo_path,
    calculate_delta_func,
    full_df=None,
    reference_df=None,
    position_kpi_map=None,
    trends_data=None,
    comparison_data=None,
    comparison_kpi_table=None,
    comparison_charts=None,
    background_image_path=None  # ← NUEVO PARÁMETRO
):
    """
    Genera el reporte PDF completo
    """
    # Crear PDF CON BACKGROUND
    pdf = WatfordPlayerReport(player_name, logo_path, background_image_path)  # ← CAMBIO
    
    # 1. Portada (ahora estilo MAX ALLEYNE)
    pdf.cover_page()
    
    # 2. Página de filtros
    pdf.filters_page(filters_data)
    
    # 3. Página de métricas
    # Determinar qué KPIs usar
    if position_kpi_map and player_position and player_position in position_kpi_map:
        selected_kpis = position_kpi_map[player_position]
    else:
        # Fallback: usar todos los disponibles en aggregated_metrics
        selected_kpis = list(aggregated_metrics.keys())
    
    # Preparar datos de KPIs con deltas
    metric_labels = {
        "pass_completion_pct": "Passes Completed %",
        "key_passes": "Key Passes",
        "aerial_duel_pct": "Aerial Duels %",
        "take_on_success_pct": "Take-Ons Success %",
        "goal_creating_actions": "Goal Creating Actions",
        "shot_creating_actions": "Shot Creating Actions",
        "shots_on_target_pct": "Shots on Target %",
        "passes_into_penalty_area": "Passes into Penalty Area",
        "carries_into_final_third": "Carries into Final Third",
        "carries_into_penalty_area": "Carries into Penalty Area",
        "goals": "Goals",
        "assists": "Assists",
        "xG": "Expected Goals (xG)",
        "xA": "Assisted Shots",
        "ps_xG": "Post-Shot xG",
        "recoveries": "Recoveries",
        "interceptions": "Interceptions",
        "clearances": "Clearances",
        "crosses": "Crosses",
        "long_pass_pct": "Long Pass %",
        "progressive_passes": "Progressive Passes",
        "totalSaves": "Saves",
        "save_pct": "Save %",
        "goals_conceded": "Goals Conceded",
        "claimsHigh": "Claims High",
        "collected": "Collected",
        "def_actions_outside_box": "Defensive Actions Outside Box",
    }
    
    kpis_data = []
    metric_scale_factors = {}
    if isinstance(filtered_df, pd.DataFrame) and "xA" in filtered_df.columns:
        xa_series = pd.to_numeric(filtered_df["xA"], errors="coerce")
        if xa_series.dropna().quantile(0.75) > 3:
            metric_scale_factors["xA"] = 100.0

    if isinstance(reference_df, pd.DataFrame):
        baseline_df = reference_df
    elif isinstance(full_df, pd.DataFrame) and not full_df.empty:
        baseline_df = full_df
    else:
        baseline_df = filtered_df
    for kpi in selected_kpis:
        if kpi in aggregated_metrics:
            value = aggregated_metrics[kpi]
            
            # Calcular delta
            delta, delta_pct = calculate_delta_func(filtered_df, baseline_df, kpi)

            scale_factor = metric_scale_factors.get(kpi, 1.0)
            if isinstance(value, (int, float, np.number)) and pd.notna(value):
                value = float(value) / scale_factor
            if isinstance(delta, (int, float, np.number)) and pd.notna(delta):
                delta = float(delta) / scale_factor
            
            # Formatear valor
            if isinstance(value, (int, float, np.number)) and pd.notna(value):
                value = float(value)
                if 'pct' in kpi or '%' in kpi:
                    formatted_value = f"{value:.1f}%"
                elif kpi in ['xG', 'xA', 'ps_xG']:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{int(round(value))}"
            else:
                formatted_value = str(value)
            
            kpis_data.append({
                'title': metric_labels.get(kpi, kpi.replace('_', ' ').title()),
                'value': formatted_value,
                'delta': delta,
                'delta_pct': delta_pct
            })
    
    pdf.metrics_page(player_info, player_position, kpis_data)
    
    # 4. Tabla de estadísticas (solo columnas relevantes para la posición)
    pdf.stats_table_page(filtered_df, selected_kpis)
    
    # 5. Trends Stats (si se proporcionan datos)
    if trends_data:
        pdf.trends_stats_page(trends_data)
    
    # 6. Player Comparison (si se proporcionan datos)
    if comparison_data is not None and not comparison_data.empty:
        pdf.player_comparison_page(comparison_data, comparison_kpi_table, comparison_charts or [])
    
    raw_pdf = pdf.output(dest='S')
    if isinstance(raw_pdf, (bytes, bytearray)):
        return bytes(raw_pdf)
    return str(raw_pdf).encode('latin-1', errors='replace')


def generate_individual_development_report(
    player_name: str,
    fecha_inicio: str,
    fecha_fin: str,
    df_actividades: pd.DataFrame,
    df_summary: pd.DataFrame = None,
    logo_path: str = "img/watford_logo.png"
):
    """
    Genera PDF para Individual Development con tablas paginadas.
    
    Parameters:
    - player_name: Nombre del jugador
    - fecha_inicio: Fecha inicio del filtro
    - fecha_fin: Fecha fin del filtro
    - df_actividades: DataFrame con todas las actividades (Entrenamientos, Meetings, Review Clips)
    - df_summary: DataFrame opcional con resumen por tipo de actividad
    - logo_path: Ruta al logo de Watford
    
    Returns:
    - bytes del PDF generado
    """
    from fpdf import FPDF
    import pandas as pd
    from datetime import datetime
    import os
    
    class IndividualDevelopmentReport(FPDF):
        def __init__(self, player_name, logo_path):
            super().__init__(orientation='P', unit='mm', format='A4')  # Portrait
            self.player_name = player_name
            self.logo_path = logo_path
            self.set_auto_page_break(auto=True, margin=15)
            
            # Colores Watford
            self.COLOR_YELLOW = (252, 236, 3)
            self.COLOR_GRAY = (136, 136, 136)
            self.COLOR_BLACK = (0, 0, 0)
            
        def header(self):
            if self.page_no() > 1:
                self.set_font('Arial', 'B', 12)
                self.set_text_color(*self.COLOR_BLACK)
                self.cell(0, 10, self.player_name, 0, 0, 'C')
                
                try:
                    if os.path.exists(self.logo_path):
                        self.image(self.logo_path, x=180, y=8, w=20)
                except:
                    pass
                
                self.ln(15)
        
        def footer(self):
            if self.page_no() > 1:
                self.set_y(-15)
                self.set_font('Arial', 'I', 10)
                self.set_text_color(*self.COLOR_GRAY)
                self.cell(0, 10, 'Individual Development Report', 0, 0, 'C')
                self.set_font('Arial', 'I', 10)
                page_text = f'Page {self.page_no() - 1}'
                self.cell(0, 10, page_text, 0, 0, 'R')
        
        def cover_page(self):
            self.add_page()
            self.ln(60)
            
            self.set_font('Arial', 'B', 36)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 20, self.player_name, 0, 1, 'C')
            
            self.set_font('Arial', '', 24)
            self.set_text_color(*self.COLOR_GRAY)
            self.cell(0, 15, 'Individual Development', 0, 1, 'C')
            
            try:
                if os.path.exists(self.logo_path):
                    logo_width = 80
                    logo_x = (210 - logo_width) / 2
                    logo_y = 120
                    self.image(self.logo_path, x=logo_x, y=logo_y, w=logo_width)
            except:
                pass
            
            self.set_y(-30)
            self.set_font('Arial', 'I', 10)
            self.set_text_color(*self.COLOR_GRAY)
            generated_date = datetime.now().strftime("%B %d, %Y")
            self.cell(0, 10, f'Generated: {generated_date}', 0, 0, 'C')
        
        def filters_page(self, fecha_inicio, fecha_fin):
            self.add_page()
            
            self.set_font('Arial', 'B', 18)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 12, 'Report Filters', 0, 1, 'L')
            self.ln(5)
            
            # Fecha inicio
            self.set_font('Arial', 'B', 12)
            self.cell(50, 10, 'Period:', 0, 0, 'L')
            self.set_font('Arial', '', 12)
            self.cell(0, 10, f'{fecha_inicio} - {fecha_fin}', 0, 1, 'L')
            self.ln(10)
        
        def summary_page(self, df_summary):
            """Página con resumen de actividades por tipo"""
            self.add_page()
            
            self.set_font('Arial', 'B', 18)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 12, 'Activities Summary', 0, 1, 'L')
            self.ln(5)
            
            if df_summary is None or df_summary.empty:
                self.set_font('Arial', 'I', 12)
                self.set_text_color(*self.COLOR_GRAY)
                self.cell(0, 10, 'No activities in this period', 0, 1, 'C')
                return
            
            # Tabla resumen
            self.set_font('Arial', 'B', 10)
            self.set_fill_color(*self.COLOR_YELLOW)
            self.set_text_color(*self.COLOR_BLACK)
            
            # Headers
            col_widths = [80, 50]
            self.cell(col_widths[0], 10, 'Activity Type', 1, 0, 'C', True)
            self.cell(col_widths[1], 10, 'Count', 1, 1, 'C', True)
            
            # Rows
            self.set_font('Arial', '', 10)
            for idx, row in df_summary.iterrows():
                fill = idx % 2 == 0
                if fill:
                    self.set_fill_color(245, 245, 245)
                
                self.cell(col_widths[0], 8, str(row.get('tipo', row.get('Activity Type', ''))), 1, 0, 'L', fill)
                self.cell(col_widths[1], 8, str(row.get('count', row.get('Count', 0))), 1, 1, 'C', fill)
        
        def activities_table_paginated(self, df_actividades):
            """
            Tabla de actividades SIN columna Description (redundante).
            Más espacio para Detail.
            """
            self.add_page()
            
            # Título de sección
            self.set_font('Arial', 'B', 18)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 12, 'Individual Activities', 0, 1, 'L')
            self.ln(3)
            
            if df_actividades.empty:
                self.set_font('Arial', 'I', 12)
                self.set_text_color(*self.COLOR_GRAY)
                self.cell(0, 10, 'No activities registered', 0, 1, 'C')
                return
            
            # Preparar DataFrame
            df_table = df_actividades.copy()
            
            # Asegurar que fecha esté formateada
            if 'fecha' in df_table.columns:
                df_table['fecha'] = pd.to_datetime(df_table['fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
            
            # ===== SELECCIONAR SOLO: Date, Type, Detail (SIN Description) =====
            display_columns = []
            if 'fecha' in df_table.columns:
                display_columns.append('fecha')
            if 'tipo' in df_table.columns:
                display_columns.append('tipo')
            if 'subtipo' in df_table.columns:
                display_columns.append('subtipo')
            elif 'titulo' in df_table.columns:
                display_columns.append('titulo')
            # ❌ NO incluir 'descripcion'
            
            df_table = df_table[display_columns].fillna('')
            
            # Renombrar columnas para headers
            column_labels = {
                'fecha': 'Date',
                'tipo': 'Type',
                'subtipo': 'Detail',
                'titulo': 'Title'
            }
            df_table = df_table.rename(columns=column_labels)
            
            # ===== Anchos de columna CON MÁS ESPACIO para Detail =====
            col_widths = {
                'Date': 30,
                'Type': 35,
                'Detail': 115,  # ← MÁS ESPACIO (antes era 50)
                'Title': 115
            }
            
            actual_widths = [col_widths.get(col, 40) for col in df_table.columns]
            
            # Función helper para imprimir headers
            def print_headers():
                self.set_fill_color(*self.COLOR_YELLOW)
                self.set_text_color(*self.COLOR_BLACK)
                self.set_font('Arial', 'B', 9)
                for col_name, width in zip(df_table.columns, actual_widths):
                    self.cell(width, 8, str(col_name), 1, 0, 'C', True)
                self.ln()
            
            # Imprimir headers iniciales
            print_headers()
            
            # Función auxiliar para calcular líneas necesarias
            def calculate_lines_needed(text, width, font_size=7):
                if not text or text == '':
                    return 1
                chars_per_line = int(width * 3.2)
                if chars_per_line <= 0:
                    return 1
                words = str(text).split()
                lines = 1
                current_line_length = 0
                for word in words:
                    word_length = len(word) + 1
                    if current_line_length + word_length > chars_per_line:
                        lines += 1
                        current_line_length = word_length
                    else:
                        current_line_length += word_length
                return lines
            
            # Procesar filas
            self.set_font('Arial', '', 7)
            self.set_text_color(*self.COLOR_BLACK)
            
            for idx, row in df_table.iterrows():
                # ===== CALCULAR ALTURA DE LA FILA =====
                max_lines = 1
                line_height = 5
                
                for col_name, width in zip(df_table.columns, actual_widths):
                    value = str(row[col_name]) if pd.notna(row[col_name]) and row[col_name] != '' else ''
                    lines_needed = calculate_lines_needed(value, width, 7)
                    max_lines = max(max_lines, lines_needed)
                
                row_height = max_lines * line_height + 2
                
                # ===== VERIFICAR SI HAY ESPACIO =====
                if self.get_y() + row_height > 260:
                    self.add_page()
                    self.set_font('Arial', 'B', 18)
                    self.set_text_color(*self.COLOR_BLACK)
                    self.cell(0, 12, 'Individual Activities (continued)', 0, 1, 'L')
                    self.ln(3)
                    print_headers()
                    self.set_font('Arial', '', 7)
                
                # ===== DIBUJAR FILA =====
                x_start = self.get_x()
                y_start = self.get_y()
                
                # Alternar color de fondo
                if idx % 2 == 0:
                    self.set_fill_color(245, 245, 245)
                    fill = True
                else:
                    fill = False
                
                # Dibujar cada celda
                for col_idx, (col_name, width) in enumerate(zip(df_table.columns, actual_widths)):
                    value = str(row[col_name]) if pd.notna(row[col_name]) and row[col_name] != '' else ''
                    x_col = x_start + sum(actual_widths[:col_idx])
                    
                    # Dibujar borde
                    self.set_xy(x_col, y_start)
                    if fill:
                        self.set_fill_color(245, 245, 245)
                        self.rect(x_col, y_start, width, row_height, 'DF')
                    else:
                        self.rect(x_col, y_start, width, row_height, 'D')
                    
                    # Escribir texto línea por línea
                    y_text = y_start + 1
                    words = value.split()
                    current_line = ""
                    chars_per_line = int((width - 2) * 3.2)
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if len(test_line) <= chars_per_line:
                            current_line = test_line
                        else:
                            self.set_xy(x_col + 1, y_text)
                            self.cell(width - 2, line_height, current_line, 0, 0, 'L')
                            y_text += line_height
                            current_line = word
                            if y_text + line_height > y_start + row_height:
                                break
                    
                    if current_line and y_text + line_height <= y_start + row_height:
                        self.set_xy(x_col + 1, y_text)
                        self.cell(width - 2, line_height, current_line, 0, 0, 'L')
                
                # Mover a siguiente fila
                self.set_xy(x_start, y_start + row_height)
    
    # ========== GENERAR PDF ==========
    pdf = IndividualDevelopmentReport(player_name, logo_path)
    
    # Portada
    pdf.cover_page()
    
    # Filtros
    pdf.filters_page(fecha_inicio, fecha_fin)
    
    # Resumen (si está disponible)
    if df_summary is not None and not df_summary.empty:
        pdf.summary_page(df_summary)
    
    # Tabla completa de actividades (PAGINADA)
    pdf.activities_table_paginated(df_actividades)
    
    # Retornar bytes del PDF
    raw_pdf = pdf.output(dest='S')
    if isinstance(raw_pdf, (bytes, bytearray)):
        return bytes(raw_pdf)
    return str(raw_pdf).encode('latin-1', errors='replace')

def generate_individual_development_report_landscape(
    player_name: str,
    fecha_inicio: str,
    fecha_fin: str,
    df_actividades: pd.DataFrame,
    df_summary: pd.DataFrame = None,
    df_ratings: pd.DataFrame = None,  # ← NUEVO: DataFrame con ratings de WhoScored
    fig_comparison: any = None,
    fig_timeline: any = None,
    logo_path: str = "img/watford_logo.png",
    background_image_path: str = "img/Watford_portada.jpg"
):
    """
    V3: Ratings de WhoScored + Nombre centrado + Timeline sin deformar
    """
    from fpdf import FPDF
    import pandas as pd
    from datetime import datetime
    import os
    import tempfile
    import plotly.io as pio
    import plotly.graph_objects as go
    
    class IndividualDevelopmentLandscape(FPDF):
        def __init__(self, player_name, logo_path, background_image_path):
            super().__init__(orientation='L', unit='mm', format='A4')
            self.player_name = player_name
            self.logo_path = logo_path
            self.background_image_path = background_image_path
            self.set_auto_page_break(auto=False)
            
            self.COLOR_YELLOW = (252, 236, 3)
            self.COLOR_RED = (237, 28, 36)
            self.COLOR_GRAY_BG = (50, 50, 50)
            self.COLOR_GRAY_TEXT = (136, 136, 136)
            self.COLOR_BLACK = (0, 0, 0)
            self.COLOR_WHITE = (255, 255, 255)
        
        def header(self):
            """Header con nombre CENTRADO VERTICALMENTE"""
            if self.page_no() > 1:
                # Fondo gris
                self.set_fill_color(*self.COLOR_GRAY_BG)
                self.rect(0, 0, 297, 210, 'F')
                
                # ===== NOMBRE CENTRADO VERTICALMENTE =====
                self.set_xy(150, 12)  # ← AJUSTADO para centrar verticalmente
                self.set_font('Arial', 'B', 26)  # ← REDUCIDO para no chocar con logo
                self.set_text_color(*self.COLOR_WHITE)
                self.cell(110, 15, self.player_name.upper(), 0, 0, 'R')  # ← Ancho reducido
                
                # Logo
                try:
                    if os.path.exists(self.logo_path):
                        self.image(self.logo_path, x=270, y=8, w=20)
                except:
                    pass
                
                self.set_y(35)
        
        def footer(self):
            if self.page_no() > 1:
                self.set_y(-15)
                self.set_font('Arial', 'I', 10)
                self.set_text_color(*self.COLOR_WHITE)
                self.cell(0, 10, 'Individual Development Report', 0, 0, 'C')
                page_text = f'Page {self.page_no() - 1}'
                self.cell(0, 10, page_text, 0, 0, 'R')
        
        def cover_page_max_alleyne_style(self):
            self.add_page()
            
            self.set_fill_color(*self.COLOR_GRAY_BG)
            self.rect(0, 0, 60, 210, 'F')
            
            self.set_fill_color(*self.COLOR_RED)
            self.rect(60, 0, 3, 210, 'F')
            
            try:
                if os.path.exists(self.logo_path):
                    self.image(self.logo_path, x=10, y=15, w=40)
            except:
                pass
            
            self.set_xy(5, 90)
            self.set_font('Arial', 'B', 14)
            self.set_text_color(*self.COLOR_WHITE)
            self.multi_cell(50, 8, 'INDIVIDUAL\nDEVELOPMENT', 0, 'C')
            
            self.set_xy(5, 180)
            self.set_font('Arial', '', 12)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(50, 8, 'ALBERTO GARRIDO', 0, 0, 'C')
            
            try:
                if os.path.exists(self.background_image_path):
                    self.image(self.background_image_path, x=63, y=0, w=234, h=210)
            except Exception as e:
                print(f"Error loading background: {e}")
            
            self.set_xy(150, 20)
            self.set_font('Arial', 'B', 32)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(130, 15, self.player_name.upper(), 0, 0, 'R')
        
        def activities_chart_and_table_page(self, df_actividades, fig_comparison_path=None):
            self.add_page()
            
            self.set_font('Arial', 'B', 20)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(0, 12, 'INDIVIDUAL ACTIVITIES', 0, 1, 'L')
            self.ln(3)
            
            if fig_comparison_path and os.path.exists(fig_comparison_path):
                try:
                    self.image(fig_comparison_path, x=15, y=50, w=250, h=80)
                    self.set_y(135)
                except Exception as e:
                    print(f"Error inserting chart: {e}")
                    self.set_y(50)
            else:
                self.set_y(50)
            
            self.ln(5)
            self._activities_table_paginated(df_actividades)
        
        def _activities_table_paginated(self, df_actividades):
            if df_actividades.empty:
                self.set_font('Arial', 'I', 12)
                self.set_text_color(*self.COLOR_WHITE)
                self.cell(0, 10, 'No activities registered', 0, 1, 'C')
                return
            
            df_table = df_actividades.copy()
            
            if 'fecha' in df_table.columns:
                df_table['fecha'] = pd.to_datetime(df_table['fecha'], errors='coerce').dt.strftime('%Y-%m-%d')
            
            display_columns = []
            if 'fecha' in df_table.columns:
                display_columns.append('fecha')
            if 'tipo' in df_table.columns:
                display_columns.append('tipo')
            if 'subtipo' in df_table.columns:
                display_columns.append('subtipo')
            
            df_table = df_table[display_columns].fillna('')
            
            column_labels = {'fecha': 'Date', 'tipo': 'Type', 'subtipo': 'Detail'}
            df_table = df_table.rename(columns=column_labels)
            
            col_widths = {'Date': 35, 'Type': 40, 'Detail': 200}
            actual_widths = [col_widths.get(col, 40) for col in df_table.columns]
            
            def print_headers():
                self.set_fill_color(*self.COLOR_YELLOW)
                self.set_text_color(*self.COLOR_BLACK)
                self.set_font('Arial', 'B', 10)
                for col_name, width in zip(df_table.columns, actual_widths):
                    self.cell(width, 8, str(col_name), 1, 0, 'C', True)
                self.ln()
            
            def calculate_lines_needed(text, width):
                if not text:
                    return 1
                chars_per_line = int(width * 3.5)
                if chars_per_line <= 0:
                    return 1
                words = str(text).split()
                lines, current_line_length = 1, 0
                for word in words:
                    word_length = len(word) + 1
                    if current_line_length + word_length > chars_per_line:
                        lines += 1
                        current_line_length = word_length
                    else:
                        current_line_length += word_length
                return lines
            
            print_headers()
            
            self.set_font('Arial', '', 8)
            self.set_text_color(*self.COLOR_BLACK)
            
            for idx, row in df_table.iterrows():
                max_lines = 1
                line_height = 5
                for col_name, width in zip(df_table.columns, actual_widths):
                    value = str(row[col_name]) if pd.notna(row[col_name]) and row[col_name] != '' else ''
                    lines_needed = calculate_lines_needed(value, width)
                    max_lines = max(max_lines, lines_needed)
                
                row_height = max_lines * line_height + 2
                
                if self.get_y() + row_height > 180:
                    self.add_page()
                    self.set_font('Arial', 'B', 20)
                    self.set_text_color(*self.COLOR_WHITE)
                    self.cell(0, 12, 'INDIVIDUAL ACTIVITIES (continued)', 0, 1, 'L')
                    self.ln(3)
                    print_headers()
                    self.set_font('Arial', '', 8)
                
                x_start, y_start = self.get_x(), self.get_y()
                fill = idx % 2 == 0
                if fill:
                    self.set_fill_color(245, 245, 245)
                else:
                    self.set_fill_color(255, 255, 255)
                
                for col_idx, (col_name, width) in enumerate(zip(df_table.columns, actual_widths)):
                    value = str(row[col_name]) if pd.notna(row[col_name]) and row[col_name] != '' else ''
                    x_col = x_start + sum(actual_widths[:col_idx])
                    
                    self.set_xy(x_col, y_start)
                    self.rect(x_col, y_start, width, row_height, 'DF')
                    
                    y_text = y_start + 1
                    words = value.split()
                    current_line = ""
                    chars_per_line = int((width - 2) * 3.5)
                    
                    for word in words:
                        test_line = current_line + (" " if current_line else "") + word
                        if len(test_line) <= chars_per_line:
                            current_line = test_line
                        else:
                            self.set_xy(x_col + 1, y_text)
                            self.cell(width - 2, line_height, current_line, 0, 0, 'L')
                            y_text += line_height
                            current_line = word
                            if y_text + line_height > y_start + row_height:
                                break
                    
                    if current_line and y_text + line_height <= y_start + row_height:
                        self.set_xy(x_col + 1, y_text)
                        self.cell(width - 2, line_height, current_line, 0, 0, 'L')
                
                self.set_xy(x_start, y_start + row_height)
        
        def timeline_page(self, fig_timeline_path=None, fig_rating_path=None):
            self.add_page()
            
            self.set_font('Arial', 'B', 22)
            self.set_text_color(*self.COLOR_WHITE)
            self.cell(0, 12, f'{self.player_name.upper()} EVOLUTION', 0, 1, 'L')
            self.ln(5)
            
            # ===== TIMELINE SIN DEFORMAR =====
            if fig_timeline_path and os.path.exists(fig_timeline_path):
                try:
                    # Ajustar tamaño para evitar deformación
                    # self.image(fig_timeline_path, x=15, y=50, w=265, h=55)  # ← AJUSTADO
                    # ✅ Timeline con mejor proporción
                    self.image(fig_timeline_path, x=15, y=50, w=265, h=70)
                    self.set_y(110)
                except Exception as e:
                    print(f"Error inserting timeline: {e}")
                    self.set_y(50)
            
            # Rating Evolution
            if fig_rating_path and os.path.exists(fig_rating_path):
                try:
                    self.ln(2)
                    self.set_font('Arial', 'B', 16)
                    self.set_text_color(*self.COLOR_WHITE)
                    self.cell(0, 10, '', 0, 1, 'L')
                    self.ln(2)
                    # Ajustar tamaño para rating
                    # self.image(fig_rating_path, x=15, y=self.get_y(), w=265, h=60)  # ← AJUSTADO
                    # ✅ Rating con mejor proporción
                    self.image(fig_rating_path, x=15, y=self.get_y(), w=265, h=75)
                except Exception as e:
                    print(f"Error inserting rating: {e}")
    
    # ========== GENERAR PDF ==========
    pdf = IndividualDevelopmentLandscape(player_name, logo_path, background_image_path)
    pdf.cover_page_max_alleyne_style()
    
    temp_files = []
    fig_comparison_path = None
    fig_timeline_path = None
    fig_rating_path = None
    
    try:
        if fig_comparison:
            temp_comparison = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_comparison_path = temp_comparison.name
            temp_comparison.close()
            try:
                pio.write_image(fig_comparison, temp_comparison_path, width=1000, height=400)
                fig_comparison_path = temp_comparison_path
                temp_files.append(temp_comparison_path)
            except Exception as e:
                print(f"Error exporting comparison chart image: {e}")
                try:
                    os.unlink(temp_comparison_path)
                except Exception:
                    pass
        
        if fig_timeline:
            temp_timeline = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_timeline_path = temp_timeline.name
            temp_timeline.close()
            try:
                # ===== REDUCIR ALTURA PARA EVITAR DEFORMACIÓN =====
                # pio.write_image(fig_timeline, temp_timeline.name, width=1200, height=350)  # ← AJUSTADO
                # ✅ MAYOR RESOLUCIÓN PARA MEJOR CALIDAD
                pio.write_image(fig_timeline, temp_timeline_path, width=1600, height=500, scale=2)
                fig_timeline_path = temp_timeline_path
                temp_files.append(temp_timeline_path)
            except Exception as e:
                print(f"Error exporting timeline chart image: {e}")
                try:
                    os.unlink(temp_timeline_path)
                except Exception:
                    pass
        
        # ===== RATING EVOLUTION DE WHOSCORED =====
        if df_ratings is not None and not df_ratings.empty:
            try:
                # Agrupar por mes
                df_ratings['year_month'] = pd.to_datetime(df_ratings['matchDate']).dt.to_period('M')
                monthly_rating = df_ratings.groupby('year_month')['ratings_clean'].mean().reset_index()
                monthly_rating['year_month_str'] = monthly_rating['year_month'].astype(str)
                
                if len(monthly_rating) > 0:
                    # ✅ FUNCIÓN PARA ASIGNAR COLOR SEGÚN RATING
                    def get_rating_color(rating):
                        if rating <= 5.0:
                            return '#FF0000'  # Rojo
                        elif 5.0 < rating <= 6.0:
                            return '#FF8C00'  # Naranja
                        elif 6.0 < rating <= 7.5:
                            return '#FFD700'  # Amarillo
                        else:  # >= 7.5
                            return '#00FF00'  # Verde
                    
                    # ✅ ASIGNAR COLORES A CADA BARRA
                    monthly_rating['color'] = monthly_rating['ratings_clean'].apply(get_rating_color)
                    
                    fig_rating = go.Figure()
                    
                    # ✅ BARRAS CON COLORES PERSONALIZADOS
                    fig_rating.add_trace(go.Bar(
                        x=monthly_rating['year_month_str'],
                        y=monthly_rating['ratings_clean'],
                        marker_color=monthly_rating['color'],  # ← COLORES PERSONALIZADOS
                        text=monthly_rating['ratings_clean'].round(2),  # ← ETIQUETAS NUMÉRICAS
                        textposition='outside',  # ← POSICIÓN ENCIMA
                        textfont=dict(size=12, color='black', family='Arial'),  # ← ESTILO ETIQUETAS
                        hovertemplate='%{x}<br>Rating: %{y:.2f}<extra></extra>'
                    ))

                    # ✅ LÍNEA DE PROMEDIO (NEGRA)
                    avg_rating = monthly_rating['ratings_clean'].mean()
                    fig_rating.add_hline(
                        y=avg_rating,
                        line_dash="dash",
                        line_color="black",  # ← CAMBIO A NEGRO
                        line_width=2,
                        annotation_text=f"Promedio: {avg_rating:.2f}",
                        annotation_position="top right",
                        annotation=dict(
                            font=dict(size=10, color="black"),
                            bgcolor="rgba(255,255,255,0.8)"
                        )
                    )
                    
                    # ✅ LÍNEA DE TENDENCIA (SOLO SI HAY 2+ MESES)
                    if len(monthly_rating) >= 2:
                        # Calcular tendencia lineal
                        x_numeric = np.arange(len(monthly_rating))
                        y_values = monthly_rating['ratings_clean'].values
                        
                        # Regresión lineal simple
                        z = np.polyfit(x_numeric, y_values, 1)
                        p = np.poly1d(z)
                        trend_line = p(x_numeric)
                        
                        # Determinar color de tendencia
                        slope = z[0]
                        if slope > 0:
                            trend_color = '#00CC00'  # Verde (tendencia positiva)
                            trend_label = 'Tendencia ↑'
                        else:
                            trend_color = '#FF0000'  # Rojo (tendencia negativa)
                            trend_label = 'Tendencia ↓'
                        
                        # Añadir línea de tendencia
                        fig_rating.add_trace(go.Scatter(
                            x=monthly_rating['year_month_str'],
                            y=trend_line,
                            mode='lines',
                            line=dict(color=trend_color, width=2, dash='dot'),
                            name=trend_label,
                            hovertemplate='Tendencia<extra></extra>',
                            showlegend=True
                        ))


                    fig_rating.update_layout(
                        title=dict(
                            text='Monthly Average Rating (WhoScored)',
                            font=dict(size=14, color='black', family='Arial Bold')
                        ),
                        xaxis_title='Month',
                        yaxis_title='Rating',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400,  # ← AUMENTADO para mejor calidad
                        font=dict(size=11),
                        yaxis=dict(
                            range=[0, 10],
                            gridcolor='lightgray',
                            gridwidth=0.5
                        ),
                        xaxis=dict(
                            gridcolor='lightgray',
                            gridwidth=0.5
                        ),
                        margin=dict(l=60, r=40, t=70, b=70),  # ← Más margen
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                            font=dict(size=10)
                        )
                    )
                    temp_rating = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    pio.write_image(fig_rating, temp_rating.name, width=1200, height=350)
                    fig_rating_path = temp_rating.name
                    temp_files.append(temp_rating.name)
            except Exception as e:
                print(f"Error creating rating chart: {e}")
        
        pdf.activities_chart_and_table_page(df_actividades, fig_comparison_path)
        pdf.timeline_page(fig_timeline_path, fig_rating_path)
        
    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    raw_pdf = pdf.output(dest='S')
    if isinstance(raw_pdf, (bytes, bytearray)):
        return bytes(raw_pdf)
    return str(raw_pdf).encode('latin-1', errors='replace')
