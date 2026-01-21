from fpdf import FPDF
import pandas as pd
from datetime import datetime
import os
import tempfile
import plotly.io as pio

class WatfordPlayerReport(FPDF):
    """
    Generador de reportes PDF para jugadores de Watford FC
    """
    
    def __init__(self, player_name, logo_path):
        super().__init__(orientation='L', unit='mm', format='A4')  # Landscape
        self.player_name = player_name
        self.logo_path = logo_path
        self.set_auto_page_break(auto=True, margin=15)
        
        # Colores Watford
        self.COLOR_YELLOW = (252, 236, 3)      # #fcec03
        self.COLOR_PINK = (255, 230, 230)       # #ffe6e6
        self.COLOR_GRAY = (136, 136, 136)       # #888888
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_WHITE = (255, 255, 255)
        
    def header(self):
        """Header personalizado: Nombre centrado + Logo derecha"""
        if self.page_no() > 1:  # No header en portada
            # Nombre del jugador (centrado)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(*self.COLOR_BLACK)
            self.cell(0, 10, self.player_name, 0, 0, 'C')
            
            # Logo a la derecha
            try:
                if os.path.exists(self.logo_path):
                    # Posición: esquina superior derecha
                    self.image(self.logo_path, x=270, y=8, w=20)
            except Exception as e:
                print(f"Error loading logo in header: {e}")
            
            self.ln(15)
    
    def footer(self):
        """Footer personalizado: Texto centrado + Número página derecha"""
        if self.page_no() > 1:  # No footer en portada
            self.set_y(-15)
            
            # "Player Stats Report" centrado
            self.set_font('Arial', 'I', 10)
            self.set_text_color(*self.COLOR_GRAY)
            self.cell(0, 10, 'Player Stats Report', 0, 0, 'C')
            
            # Número de página a la derecha
            self.set_font('Arial', 'I', 10)
            page_text = f'Page {self.page_no() - 1}'  # -1 porque portada no cuenta
            self.cell(0, 10, page_text, 0, 0, 'R')
    
    def cover_page(self):
        """
        Portada con:
        - Título: Nombre del jugador
        - Subtítulo: Player Stats Report
        - Medio logo del Watford (parte derecha visible)
        """
        self.add_page()
        
        # Espacio superior
        self.ln(60)
        
        # Título: Nombre del jugador
        self.set_font('Arial', 'B', 36)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(0, 20, self.player_name, 0, 1, 'C')
        
        # Subtítulo
        self.set_font('Arial', '', 24)
        self.set_text_color(*self.COLOR_GRAY)
        self.cell(0, 15, 'Player Stats Report', 0, 1, 'C')
        
        # Logo grande (mitad derecha visible) - centrado verticalmente
        try:
            if os.path.exists(self.logo_path):
                # Calcular posición para que el borde izquierdo del logo esté en el límite izquierdo
                # pero solo se vea la mitad derecha
                logo_width = 120  # Ancho total del logo
                logo_x = -logo_width / 2  # Mitad fuera del papel
                logo_y = 90  # Centrado verticalmente (antes era 130)
                
                self.image(self.logo_path, x=logo_x, y=logo_y, w=logo_width)
        except Exception as e:
            print(f"Error loading cover logo: {e}")
        
        # Fecha de generación
        self.set_y(-30)
        self.set_font('Arial', 'I', 10)
        self.set_text_color(*self.COLOR_GRAY)
        generated_date = datetime.now().strftime("%B %d, %Y")
        self.cell(0, 10, f'Generated: {generated_date}', 0, 0, 'C')
    
    def filters_page(self, filters_data):
        """
        Página 2: Filtros aplicados
        - Season
        - Date Range
        - Matches seleccionados
        """
        self.add_page()
        
        # Título
        self.set_font('Arial', 'B', 20)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(0, 15, 'Filters Applied', 0, 1, 'L')
        self.ln(5)
        
        # Season
        self.set_font('Arial', 'B', 14)
        self.cell(60, 10, 'Season:', 0, 0, 'L')
        self.set_font('Arial', '', 14)
        season = filters_data.get('season', 'All seasons')
        # Si es None, cambiar a "All seasons"
        if season is None or str(season).lower() == 'none':
            season = 'All seasons'
        self.cell(0, 10, str(season), 0, 1, 'L')
        self.ln(3)
        
        # Date Range
        self.set_font('Arial', 'B', 14)
        self.cell(60, 10, 'Date Range:', 0, 0, 'L')
        self.set_font('Arial', '', 14)
        start = filters_data.get('start_date', 'N/A')
        end = filters_data.get('end_date', 'N/A')
        if isinstance(start, pd.Timestamp):
            start = start.strftime('%Y-%m-%d')
        if isinstance(end, pd.Timestamp):
            end = end.strftime('%Y-%m-%d')
        self.cell(0, 10, f'{start} to {end}', 0, 1, 'L')
        self.ln(8)
        
        # Selected Matches
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Selected Matches:', 0, 1, 'L')
        self.ln(3)
        
        matches = filters_data.get('selected_matches', [])
        if matches:
            self.set_font('Arial', '', 11)
            
            # Crear tabla de matches (2 columnas)
            col_width = 135
            row_height = 8
            
            for i, match in enumerate(matches):
                # Limitar a primeros 20 partidos para no saturar
                if i >= 20:
                    self.set_font('Arial', 'I', 10)
                    self.cell(0, 8, f'... and {len(matches) - 20} more matches', 0, 1, 'L')
                    break
                
                # Alternar columnas
                if i % 2 == 0:
                    x_pos = self.get_x()
                    y_pos = self.get_y()
                
                # Dibujar celda con borde
                self.set_fill_color(245, 245, 245)
                self.cell(col_width, row_height, f'  {match}', 1, 0, 'L', True)
                
                if i % 2 == 0:
                    # Primera columna - no saltar línea
                    self.set_xy(x_pos + col_width + 2, y_pos)
                else:
                    # Segunda columna - saltar línea
                    self.ln()
            
            # Si terminó en columna 1, saltar línea
            if len(matches) % 2 == 1:
                self.ln()
        else:
            self.set_font('Arial', 'I', 12)
            self.set_text_color(*self.COLOR_GRAY)
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
            arrow = chr(9650)  # ▲
        elif delta < 0:
            arrow = chr(9660)  # ▼
        else:
            arrow = ''
        
        delta_text = f'{arrow} {delta:+.1f} ({delta_pct:+.1f}%)'
        self.cell(width, 4, delta_text, 0, 0, 'C')
    
    def metrics_page(self, player_info, player_position, kpis_data):
        """
        Página 3: Info del jugador + Métricas KPI
        - 7 cards de info personal (rosa)
        - Cards de métricas por posición (amarillas)
        """
        self.add_page()
        
        # Nombre del jugador como título
        self.set_font('Arial', 'B', 18)
        self.set_text_color(*self.COLOR_BLACK)
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
        info_keys = ['age', 'shirtNo', 'height', 'weight', 'games_played', 
                     'games_as_starter', 'total_minutes']
        
        for i, (label, key) in enumerate(zip(info_labels, info_keys)):
            x = start_x + (card_width + spacing) * i
            value = player_info.get(key, 'N/A')
            if key == 'total_minutes':
                value = int(value) if value != 'N/A' else 'N/A'
            self.draw_info_card(x, start_y, card_width, card_height, label, value)
        
        self.set_y(start_y + card_height + 10)
        
        # "Showing Metrics for position: X"
        self.set_font('Arial', 'I', 11)
        self.set_text_color(*self.COLOR_GRAY)
        self.cell(0, 8, f'Showing Metrics for position: {player_position}', 0, 1, 'L')
        self.ln(1)  # Reducido de 3 a 1
        
        # Cards de métricas KPI (4 por fila)
        kpi_card_width = 68
        kpi_card_height = 26  # Reducido de 28 a 26
        kpi_spacing = 3
        kpis_per_row = 4
        
        kpi_start_x = 15
        kpi_start_y = self.get_y()
        
        for i, kpi_item in enumerate(kpis_data):
            row = i // kpis_per_row
            col = i % kpis_per_row
            
            x = kpi_start_x + (kpi_card_width + kpi_spacing) * col
            y = kpi_start_y + (kpi_card_height + kpi_spacing) * row
            
            self.draw_metric_card(
                x, y, kpi_card_width, kpi_card_height,
                kpi_item['title'],
                kpi_item['value'],
                kpi_item['delta'],
                kpi_item['delta_pct']
            )
    
    def stats_table_page(self, df, position_kpis):
        """
        Página 4: Tabla de estadísticas
        Solo columnas relevantes para la posición
        """
        self.add_page()
        
        # Subtítulo "Player Stats" (igual estructura que página 3 con nombre)
        self.set_font('Arial', 'B', 18)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(0, 12, 'Player Stats', 0, 1, 'L')
        self.ln(3)
        
        if df.empty:
            self.set_font('Arial', 'I', 12)
            self.set_text_color(*self.COLOR_GRAY)
            self.cell(0, 10, 'No data available', 0, 1, 'C')
            return
        
        # Columnas a mostrar: fecha, oponente + KPIs de posición
        base_cols = ['matchDate', 'oppositionTeamName']
        
        # Filtrar solo columnas que existen en el df
        available_kpis = [col for col in position_kpis if col in df.columns]
        table_columns = base_cols + available_kpis
        
        # Filtrar df
        df_table = df[table_columns].copy()
        
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
            "xA": "xA",
        }
        
        for col in available_kpis:
            if col in metric_labels:
                column_labels[col] = metric_labels[col]
        
        df_table = df_table.rename(columns=column_labels)
        
        # Configurar tabla
        self.set_font('Arial', '', 7)
        
        # Calcular anchos de columna (ajustados a landscape)
        total_width = 277  # Ancho disponible en landscape
        date_width = 25
        opponent_width = 40
        remaining_width = total_width - date_width - opponent_width
        metric_width = remaining_width / len(available_kpis) if available_kpis else 20
        
        col_widths = [date_width, opponent_width] + [metric_width] * len(available_kpis)
        
        # Headers
        self.set_fill_color(*self.COLOR_YELLOW)
        self.set_text_color(*self.COLOR_BLACK)
        self.set_font('Arial', 'B', 7)
        
        for col_name, width in zip(df_table.columns, col_widths):
            self.cell(width, 8, str(col_name), 1, 0, 'C', True)
        self.ln()
        
        # Rows
        self.set_font('Arial', '', 6)
        self.set_text_color(*self.COLOR_BLACK)
        
        for idx, row in df_table.iterrows():
            # Alternar color de fondo
            if idx % 2 == 0:
                self.set_fill_color(245, 245, 245)
                fill = True
            else:
                fill = False
            
            for col_name, width in zip(df_table.columns, col_widths):
                value = row[col_name]
                # Truncar texto largo
                if isinstance(value, str) and len(value) > 15:
                    value = value[:12] + '...'
                self.cell(width, 6, str(value), 1, 0, 'C', fill)
            self.ln()
            
            # Pagination: nueva página si nos quedamos sin espacio
            if self.get_y() > 175:  # Reducido de 180 a 175 para más margen
                self.add_page()
                
                # ✅ Re-imprimir subtítulo "Player Stats" en cada página
                self.set_font('Arial', 'B', 18)
                self.set_text_color(*self.COLOR_BLACK)
                self.cell(0, 12, 'Player Stats', 0, 1, 'L')
                self.ln(3)
                
                # Re-imprimir headers de tabla
                self.set_fill_color(*self.COLOR_YELLOW)
                self.set_text_color(*self.COLOR_BLACK)
                self.set_font('Arial', 'B', 7)
                for col_name, width in zip(df_table.columns, col_widths):
                    self.cell(width, 8, str(col_name), 1, 0, 'C', True)
                self.ln()
                self.set_font('Arial', '', 6)


    def trends_stats_page(self, trends_data):
        """
        Páginas de Trends Stats: Gráficos de evolución por partido
        
        Args:
            trends_data: Lista de dicts con {
                'kpi_name': str,
                'fig': plotly figure object
            }
        """
        if not trends_data:
            return
        
        self.add_page()
        
        # Título de la sección
        self.set_font('Arial', 'B', 18)
        self.set_text_color(*self.COLOR_BLACK)
        self.cell(0, 12, 'Trends Stats', 0, 1, 'L')
        self.ln(5)
        
        # Subtítulo
        self.set_font('Arial', 'I', 12)
        self.set_text_color(*self.COLOR_GRAY)
        self.cell(0, 8, 'Performance Trends Over Time', 0, 1, 'L')
        self.ln(5)
        
        charts_per_page = 2  # 2 gráficos por página
        chart_count = 0
        
        for item in trends_data:
            # Nueva página si ya tenemos 2 gráficos
            if chart_count > 0 and chart_count % charts_per_page == 0:
                self.add_page()
                
                # Re-imprimir título en cada página
                self.set_font('Arial', 'B', 18)
                self.set_text_color(*self.COLOR_BLACK)
                self.cell(0, 12, 'Trends Stats', 0, 1, 'L')
                self.ln(5)
            
            try:
                # Guardar gráfico como imagen temporal
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # Convertir Plotly a imagen
                pio.write_image(item['fig'], tmp_path, width=1400, height=500, scale=2)
                
                # Título del gráfico
                self.set_font('Arial', 'B', 11)
                self.set_text_color(*self.COLOR_BLACK)
                self.cell(0, 8, item['kpi_name'], 0, 1, 'L')
                self.ln(2)
                
                # Insertar imagen (más grande para mejor legibilidad)
                img_width = 270  # Casi todo el ancho disponible
                img_height = 95  # Más alto
                
                self.image(tmp_path, x=10, y=self.get_y(), w=img_width, h=img_height)
                self.ln(img_height + 8)  # Más espacio entre gráficos
                
                # Eliminar archivo temporal
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
                chart_count += 1
                
            except Exception as e:
                print(f"Error adding trend chart: {e}")
                # Si falla, poner texto placeholder
                self.set_font('Arial', 'I', 10)
                self.set_text_color(*self.COLOR_GRAY)
                self.cell(0, 8, f"[Chart: {item['kpi_name']} - Could not render]", 0, 1, 'L')
                self.ln(10)


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
        self.set_text_color(*self.COLOR_BLACK)
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
                'xA': 'xA',
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
        
        # ========== GRÁFICOS COMPARATIVOS ==========
        if comparison_charts:
            charts_per_page = 2
            chart_count = 0
            
            for item in comparison_charts:
                # Nueva página si ya tenemos 2 gráficos (y no es el primero)
                if chart_count > 0 and chart_count % charts_per_page == 0:
                    self.add_page()
                    
                    # Re-imprimir título
                    self.set_font('Arial', 'B', 18)
                    self.set_text_color(*self.COLOR_BLACK)
                    self.cell(0, 12, 'Player Comparison', 0, 1, 'L')
                    self.ln(5)
                
                try:
                    # Guardar gráfico como imagen temporal
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    # Convertir Plotly a imagen (más grande que antes)
                    pio.write_image(item['fig'], tmp_path, width=1400, height=500, scale=2)
                    
                    # Título del gráfico
                    self.set_font('Arial', 'B', 11)
                    self.set_text_color(*self.COLOR_BLACK)
                    self.cell(0, 8, item['kpi_name'], 0, 1, 'L')
                    self.ln(2)
                    
                    # Insertar imagen
                    img_width = 270
                    img_height = 95
                    
                    self.image(tmp_path, x=10, y=self.get_y(), w=img_width, h=img_height)
                    self.ln(img_height + 8)
                    
                    # Eliminar archivo temporal
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
                    chart_count += 1
                    
                except Exception as e:
                    print(f"Error adding comparison chart: {e}")
                    self.set_font('Arial', 'I', 10)
                    self.set_text_color(*self.COLOR_GRAY)
                    self.cell(0, 8, f"[Chart: {item['kpi_name']} - Could not render]", 0, 1, 'L')
                    self.ln(10)


def generate_player_report(
    player_name,
    player_info,
    player_position,
    aggregated_metrics,
    filtered_df,
    filters_data,
    logo_path,
    calculate_delta_func,
    position_kpi_map=None,
    trends_data=None,
    comparison_data=None,
    comparison_kpi_table=None,
    comparison_charts=None
):
    """
    Genera el reporte PDF completo
    
    Args:
        player_name: Nombre del jugador
        player_info: Dict con info personal (age, height, etc.)
        player_position: Posición del jugador
        aggregated_metrics: Dict con valores de métricas
        filtered_df: DataFrame con datos filtrados
        filters_data: Dict con filtros aplicados (season, dates, matches)
        logo_path: Ruta al logo de Watford
        calculate_delta_func: Función para calcular deltas
        position_kpi_map: (Opcional) Dict con KPIs por posición. Si se proporciona, se usan automáticamente los KPIs de la posición
        trends_data: (Opcional) Lista de dicts con gráficos de tendencias
        comparison_data: (Opcional) DataFrame con datos generales (Players Summary)
        comparison_kpi_table: (Opcional) DataFrame con KPIs comparados (playerId, playerName, teamName + KPIs)
        comparison_charts: (Opcional) Lista de dicts con gráficos comparativos
    
    Returns:
        bytes: PDF en formato bytes
    """
    # Crear PDF
    pdf = WatfordPlayerReport(player_name, logo_path)
    
    # 1. Portada
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
        "xA": "Expected Assists (xA)",
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
    for kpi in selected_kpis:
        if kpi in aggregated_metrics:
            value = aggregated_metrics[kpi]
            
            # Calcular delta
            delta, delta_pct = calculate_delta_func(filtered_df, filtered_df, kpi)
            
            # Formatear valor
            if isinstance(value, float):
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
    
    # Generar bytes
    pdf_output = pdf.output(dest='S').encode('latin-1')
    
    return pdf_output