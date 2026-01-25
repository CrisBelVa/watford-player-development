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
        self.cell(0, 10, f'Page {self.page_no()} | Technical Documentation - Watford FC', 0, 0, 'C')

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

    # 1. Project Overview
    pdf.chapter_title(1, 'Project Overview')
    overview_text = (
        "The Watford Player Development Hub is a comprehensive sports analytics platform designed to track, "
        "manage, and optimize individual player development within Watford Football Club. By integrating match "
        "performance data with individual training activities, the platform provides a 360-degree view of player "
        "progress, enabling data-driven decision-making for both technical staff and players.\n\n"
        "Core Problems Solved:\n"
        "- Centralization of performance and development data.\n"
        "- Identification of position-specific KPIs.\n"
        "- Long-term development tracking independent of personnel changes."
    )
    pdf.chapter_body(overview_text)

    # 2. Technology Stack
    pdf.chapter_title(2, 'Technology Stack & Technical Approach')
    tech_text = (
        "The platform is built as a modular Python-based web application using Streamlit. "
        "It connects to a centralized MySQL database (hosted on AWS RDS) and employs a multi-layered architecture.\n\n"
        "Primary Technologies:\n"
        "- Python: Core language for data processing and logic.\n"
        "- Streamlit: Web application framework.\n"
        "- MySQL (AWS RDS): Relational database storage.\n"
        "- SQLAlchemy: Database abstraction and ORM.\n"
        "- Pandas / NumPy: Data manipulation and numerical computation.\n"
        "- Altair / Plotly / Matplotlib: Interactive visualizations.\n"
        "- FPDF: PDF generation for reports."
    )
    pdf.chapter_body(tech_text)

    # 3. Data Sources & Data Flow
    pdf.chapter_title(3, 'Data Sources & Data Flow')
    data_flow_text = (
        "The platform ingests data from match performance systems (match-level and event-level) "
        "and development logs (training, meetings, video reviews).\n\n"
        "Data Flow Steps:\n"
        "1. Authentication: Identifies user role and associated player ID.\n"
        "2. Query Execution: Fetches relevant data via SQLAlchemy.\n"
        "3. Normalization: Pandas handles cleaning and position mapping.\n"
        "4. Calculation: Business logic layer computes KPIs and advanced metrics (xG, etc.).\n"
        "5. Rendering: Visualizations are generated for the user interface."
    )
    pdf.chapter_body(data_flow_text)

    # 4. Database Schema
    pdf.chapter_title(4, 'Database Tables & Schema Explanation')
    schema_header = ['Table Name', 'Purpose', 'Key Columns']
    schema_data = [
        ['player_data', 'Static info & appearances', 'playerId, matchId'],
        ['player_stats', 'Match-level statistics', 'matchId, playerId'],
        ['event_data', 'Granular event actions', 'matchId, x, y, typeId'],
        ['match_data', 'Match metadata', 'matchId, startDate'],
        ['jugadores', 'Master player list', 'id, nombre, activo'],
        ['meetings', 'Staff-player meetings', 'jugador_id, fecha']
    ]
    pdf.add_table(schema_header, schema_data)

    # 5. Authentication
    pdf.chapter_title(5, 'Authentication & Login Logic')
    auth_text = (
        "The system employs a custom login mechanism:\n"
        "- Staff Login: Authenticates against staff_users.csv using username/password.\n"
        "- Player Login: Name selection + master password verification.\n"
        "- Session Handling: Uses st.session_state for persistence.\n"
        "- Security: Role-based access control ensures data privacy."
    )
    pdf.chapter_body(auth_text)

    # 6. Staff Section
    pdf.chapter_title(6, 'Staff Login Section')
    staff_text = (
        "Purpose: High-level oversight and administrative management.\n\n"
        "Key Sections:\n"
        "- General Dashboard: Overview of total activities and participation rates.\n"
        "- Individual Development: Detailed player activity history.\n"
        "- Manage Players: Activation/deactivation and database syncing.\n"
        "- Files: Excel import and validation for development logs."
    )
    pdf.chapter_body(staff_text)

    # 7. Player Section
    pdf.chapter_title(7, 'Player Login Section')
    player_text = (
        "Purpose: Self-awareness and performance benchmarking.\n\n"
        "Key Sections:\n"
        "- Overview Stats: Position-specific KPIs with deltas.\n"
        "- Trends Stats: Visual performance history. Data is sorted chronologically by match date. "
        "A red dashed line represents the Season Average (mean of the KPI across all matches).\n"
        "- Player Comparison: Benchmarking against top performers. The system selects the top 5 teams "
        "in the competition by points and extracts the top 5 players in the same position by total rating.\n"
        "- Player Details: Full match-by-match breakdown."
    )
    pdf.chapter_body(player_text)

    # 8. Metrics Breakdown
    pdf.chapter_title(8, 'Metrics - Technical Breakdown')
    metrics_text = (
        "Calculation Logic:\n"
        "- Deltas: (Filtered Average - Season Average).\n"
        "- Weighted Percentages: Calculated from raw counts (e.g., total accurate / total attempted).\n"
        "- Advanced Metrics: Heuristic xG based on shot coordinates; Progressive actions based on distance moved.\n\n"
        "Trend Stats & Comparison Logic:\n"
        "- Trend Baseline: Red dashed line = (Sum of Metric Values / Total Matches).\n"
        "- Comparison Selection: Top 5 teams by points -> Filter by position -> Rank by total_rating."
    )
    pdf.chapter_body(metrics_text)

    # 9. Design Principles
    pdf.chapter_title(9, 'Design & UX Principles')
    design_text = (
        "- KPI Prioritization: Metrics are filtered by position to reduce cognitive load.\n"
        "- Consistency: Unified visual language across all dashboards.\n"
        "- Accessibility: Clear charts and tables for non-technical users."
    )
    pdf.chapter_body(design_text)

    # 10. Limitations
    pdf.chapter_title(10, 'Limitations & Future Improvements')
    limit_text = (
        "Known Constraints:\n"
        "- Manual Excel imports for development logs.\n"
        "- Dependency on source data refresh rates.\n\n"
        "Future Roadmap:\n"
        "- Automated API integration with tracking providers.\n"
        "- Predictive analytics for development trajectories.\n"
        "- Mobile-first interface for activity logging."
    )
    pdf.chapter_body(limit_text)

    # 11. Position-Specific KPI Breakdown
    pdf.chapter_title(11, 'Position-Specific KPI Breakdown')
    
    # GK Table
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.1 Goalkeepers', 0, 1, 'L')
    gk_header = ['KPI', 'Definition', 'Calculation']
    gk_data = [
        ['totalSaves', 'Total Saves', 'Count of shots saved'],
        ['save_pct', 'Save %', '(Saves / SOT Faced) * 100'],
        ['goals_conceded', 'Goals Conceded', 'Total goals allowed'],
        ['claimsHigh', 'High Claims', 'High crosses caught'],
        ['collected', 'Collected Balls', 'Loose balls gathered'],
        ['def_actions_outside_box', 'Sweeper Actions', 'Def. actions outside box'],
        ['ps_xG', 'Post-Shot xG', 'Evaluates shot placement']
    ]
    pdf.add_table(gk_header, gk_data)

    # DF Table
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.2 Defenders (Fullbacks & Center Backs)', 0, 1, 'L')
    df_header = ['KPI', 'Definition', 'Calculation']
    df_data = [
        ['interceptions', 'Interceptions', 'Opponent passes cut off'],
        ['progressive_passes', 'Prog. Passes', '10+ yards towards goal'],
        ['recoveries', 'Ball Recoveries', 'Gaining loose ball possession'],
        ['crosses', 'Crosses', 'Balls into penalty area'],
        ['take_on_success_pct', 'Dribble Success %', '(Succ / Total) * 100'],
        ['pass_completion_pct', 'Pass Accuracy %', '(Succ / Total) * 100'],
        ['clearances', 'Clearances', 'Removing ball from danger'],
        ['long_pass_pct', 'Long Pass Acc %', '(Succ / Total) * 100'],
        ['aerial_duel_pct', 'Aerial Win %', '(Won / Total) * 100']
    ]
    pdf.add_table(df_header, df_data)

    # MF Table
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.3 Midfielders', 0, 1, 'L')
    mf_header = ['KPI', 'Definition', 'Calculation']
    mf_data = [
        ['recoveries', 'Ball Recoveries', 'Gaining loose ball possession'],
        ['interceptions', 'Interceptions', 'Opponent passes cut off'],
        ['pass_completion_pct', 'Pass Accuracy %', '(Succ / Total) * 100'],
        ['progressive_passes', 'Prog. Passes', '10+ yards towards goal'],
        ['key_passes', 'Key Passes', 'Passes leading to a shot'],
        ['passes_into_penalty_area', 'Entry Passes', 'Passes into opponent box'],
        ['goal_creating_actions', 'Goal Creation', 'Actions leading to a goal'],
        ['shot_creating_actions', 'Shot Creation', 'Actions leading to a shot']
    ]
    pdf.add_table(mf_header, mf_data)

    # FW Table
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 7, '11.4 Forwards (Wingers & Strikers)', 0, 1, 'L')
    fw_header = ['KPI', 'Definition', 'Calculation']
    fw_data = [
        ['goals', 'Goals', 'Number of goals scored'],
        ['assists', 'Assists', 'Number of assists provided'],
        ['xG', 'Expected Goals', 'Prob. of shot being a goal'],
        ['xA', 'Expected Assists', 'Prob. of pass being a goal'],
        ['shots_on_target_pct', 'Shot Accuracy %', '(SOT / Total) * 100'],
        ['carries_into_box', 'Box Carries', 'Carrying ball into box'],
        ['take_on_success_pct', 'Dribble Success %', '(Succ / Total) * 100'],
        ['goal_creating_actions', 'Goal Creation', 'Actions leading to a goal']
    ]
    pdf.add_table(fw_header, fw_data)

    output_path = '/Users/cristhiancamilobeltranvalencia/Documents/watford-player-development/Project_Technical_Documentation.pdf'
    pdf.output(output_path)
    print(f"PDF generated successfully at: {output_path}")

if __name__ == "__main__":
    generate_pdf()
