import uuid
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from player_ids import normalize_whoscored_player_id

FOCUS_CARDS_SHEET = "Player Focus Cards"
FOCUS_CARD_COLUMNS = [
    "card_id",
    "player_id",
    "player_name",
    "focus_type",
    "title",
    "description",
    "metric_keys",
    "status",
    "created_at",
    "updated_at",
    "created_by",
    "updated_by",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text


def _normalize_player_id(value: Any) -> str:
    raw = _clean_text(value)
    if not raw:
        return ""
    normalized = normalize_whoscored_player_id(raw)
    return normalized or raw


def _normalize_focus_type(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"strength", "fortaleza", "fortalezas"}:
        return "Strength"
    return "Weakness"


def _normalize_focus_status(value: Any) -> str:
    token = _clean_text(value).lower()
    if token in {"archived", "archive", "inactive", "inactivo"}:
        return "archived"
    return "active"


def parse_metric_keys(raw_value: Any) -> List[str]:
    raw = _clean_text(raw_value)
    if not raw:
        return []

    normalized = (
        raw.replace(";", "|")
        .replace(",", "|")
        .replace("/", "|")
    )
    metric_keys = []
    for metric in normalized.split("|"):
        cleaned = _clean_text(metric)
        if cleaned and cleaned not in metric_keys:
            metric_keys.append(cleaned)
    return metric_keys


def _serialize_metric_keys(metric_keys: List[str]) -> str:
    cleaned = []
    for metric in metric_keys:
        token = _clean_text(metric)
        if token and token not in cleaned:
            cleaned.append(token)
    return "|".join(cleaned)


def _now_timestamp() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_focus_cards_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FOCUS_CARD_COLUMNS)

    out = df.copy()
    for col in FOCUS_CARD_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out = out[FOCUS_CARD_COLUMNS].copy()
    for col in out.columns:
        out[col] = out[col].apply(_clean_text)

    out["player_id"] = out["player_id"].apply(_normalize_player_id)
    out["focus_type"] = out["focus_type"].apply(_normalize_focus_type)
    out["status"] = out["status"].apply(_normalize_focus_status)
    out["metric_keys"] = out["metric_keys"].apply(
        lambda value: _serialize_metric_keys(parse_metric_keys(value))
    )

    now_ts = _now_timestamp()
    for idx in out.index:
        if not _clean_text(out.at[idx, "card_id"]):
            out.at[idx, "card_id"] = uuid.uuid4().hex
        if not _clean_text(out.at[idx, "created_at"]):
            out.at[idx, "created_at"] = out.at[idx, "updated_at"] or now_ts
        if not _clean_text(out.at[idx, "updated_at"]):
            out.at[idx, "updated_at"] = out.at[idx, "created_at"] or now_ts

    out = out[out["player_id"] != ""].copy().reset_index(drop=True)
    return out


def load_focus_cards_df(*, sheets_client: Any, local_file: Path) -> pd.DataFrame:
    df = None
    if sheets_client is not None and hasattr(sheets_client, "is_configured") and sheets_client.is_configured():
        try:
            df = sheets_client.read_sheet_df(
                FOCUS_CARDS_SHEET,
                expected_columns=FOCUS_CARD_COLUMNS,
            )
        except Exception as exc:
            st.warning(
                f"Could not read Google Sheets tab '{FOCUS_CARDS_SHEET}'. Using local fallback. ({exc})"
            )

    if df is None:
        if local_file.exists():
            try:
                df = pd.read_csv(local_file, dtype=str)
            except Exception as exc:
                st.warning(f"Could not read local focus cards file. Starting empty. ({exc})")
                df = pd.DataFrame(columns=FOCUS_CARD_COLUMNS)
        else:
            df = pd.DataFrame(columns=FOCUS_CARD_COLUMNS)

    return normalize_focus_cards_df(df)


def save_focus_cards_df(*, df: pd.DataFrame, sheets_client: Any, local_file: Path) -> str:
    normalized_df = normalize_focus_cards_df(df)

    if sheets_client is not None and hasattr(sheets_client, "is_configured") and sheets_client.is_configured():
        try:
            sheets_client.write_sheet_df(FOCUS_CARDS_SHEET, normalized_df)
            return f"Google Sheets / {FOCUS_CARDS_SHEET}"
        except Exception as exc:
            st.warning(
                f"Could not save to Google Sheets ({FOCUS_CARDS_SHEET}). Saving locally. ({exc})"
            )

    local_file.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_csv(local_file, index=False)
    return str(local_file)


def _format_metric_name(metric_key: str, metric_labels: Dict[str, str]) -> str:
    return metric_labels.get(metric_key, metric_key.replace("_", " ").title())


def _card_option_label(card_row: pd.Series) -> str:
    title = _clean_text(card_row.get("title")) or "Untitled card"
    focus_type = _clean_text(card_row.get("focus_type")) or "Focus"
    status = _clean_text(card_row.get("status")) or "active"
    return f"{focus_type} | {title} [{status}]"


def _render_focus_group(
    *,
    title: str,
    cards_df: pd.DataFrame,
    metric_labels: Dict[str, str],
) -> None:
    st.markdown(f"#### {title}")
    if cards_df.empty:
        st.caption("No active cards yet.")
        return

    for _, row in cards_df.iterrows():
        metric_keys = parse_metric_keys(row.get("metric_keys"))
        metric_text = ", ".join(_format_metric_name(metric, metric_labels) for metric in metric_keys)

        with st.container(border=True):
            st.markdown(f"**{_clean_text(row.get('title')) or 'Untitled card'}**")
            description = _clean_text(row.get("description"))
            if description:
                st.caption(description)
            st.caption(f"Linked metrics: {metric_text or 'None'}")
            updated_at = _clean_text(row.get("updated_at"))
            if updated_at:
                st.caption(f"Updated: {updated_at}")


def _build_progress_rows(
    *,
    chart_df: pd.DataFrame,
    selected_metric_keys: List[str],
    metric_labels: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for metric_key in selected_metric_keys:
        metric_series = (
            chart_df[chart_df["metric_key"] == metric_key]
            .sort_values("matchDate")["value"]
            .dropna()
        )
        if metric_series.empty:
            continue

        window_size = min(3, len(metric_series))
        first_avg = float(metric_series.head(window_size).mean())
        last_avg = float(metric_series.tail(window_size).mean())
        rows.append(
            {
                "Metric": _format_metric_name(metric_key, metric_labels),
                f"Start avg ({window_size})": round(first_avg, 2),
                f"End avg ({window_size})": round(last_avg, 2),
                "Change": round(last_avg - first_avg, 2),
            }
        )
    return pd.DataFrame(rows)


def _is_lower_better_metric(metric_key: str, metric_labels: Dict[str, str]) -> bool:
    text = f"{metric_key} {_format_metric_name(metric_key, metric_labels)}".lower()
    lower_better_tokens = [
        "against",
        "conceded",
        "lost",
        "loss",
        "unsuccessful",
        "failed",
        "turnover",
        "dispossessed",
        "error",
        "foul",
        "card",
        "yellow",
        "red",
        "offside",
        "miscontrol",
        "misplaced",
    ]
    return any(token in text for token in lower_better_tokens)


def _normalize_metric_scores(
    chart_df: pd.DataFrame,
    metric_labels: Dict[str, str],
) -> pd.DataFrame:
    scored_df = chart_df.copy()
    scored_df["score"] = np.nan

    for metric_key, metric_df in scored_df.groupby("metric_key"):
        values = pd.to_numeric(metric_df["value"], errors="coerce")
        min_value = values.min()
        max_value = values.max()
        if pd.isna(min_value) or pd.isna(max_value):
            continue

        if np.isclose(max_value, min_value):
            normalized = pd.Series(50.0, index=metric_df.index)
        else:
            normalized = ((values - min_value) / (max_value - min_value)) * 100.0

        if _is_lower_better_metric(metric_key, metric_labels):
            normalized = 100.0 - normalized

        scored_df.loc[metric_df.index, "score"] = normalized.clip(0, 100)

    return scored_df


def _build_monthly_focus_df(
    chart_df: pd.DataFrame,
    metric_type_map: Dict[str, str],
    percentage_formula_map: Dict[str, tuple],
    use_per90: bool,
) -> pd.DataFrame:
    monthly_base = chart_df.copy()
    monthly_base["month_start"] = monthly_base["matchDate"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for (metric_key, metric_name, month_start), month_df in monthly_base.groupby(
        ["metric_key", "Metric", "month_start"],
        dropna=False,
    ):
        metric_type = metric_type_map.get(metric_key)
        raw_values = pd.to_numeric(
            month_df.get("raw_value", month_df["value"]),
            errors="coerce",
        )
        value = np.nan
        calculation = "Monthly average"

        if metric_type == "percentage" and metric_key in percentage_formula_map:
            numerator_values = pd.to_numeric(
                month_df.get("numerator_value", pd.Series(index=month_df.index, dtype=float)),
                errors="coerce",
            )
            denominator_values = pd.to_numeric(
                month_df.get("denominator_value", pd.Series(index=month_df.index, dtype=float)),
                errors="coerce",
            )
            denominator_sum = denominator_values.sum(min_count=1)
            numerator_sum = numerator_values.sum(min_count=1)
            if pd.notna(denominator_sum) and denominator_sum > 0:
                value = (numerator_sum / denominator_sum) * 100.0
                calculation = "Weighted by attempts"
            else:
                value = pd.to_numeric(month_df["value"], errors="coerce").mean()
                calculation = "Monthly average"
        elif use_per90:
            minutes = pd.to_numeric(
                month_df.get("minutes_reference", pd.Series(index=month_df.index, dtype=float)),
                errors="coerce",
            )
            minutes_sum = minutes.sum(min_count=1)
            raw_sum = raw_values.sum(min_count=1)
            if pd.notna(minutes_sum) and minutes_sum > 0:
                value = (raw_sum / minutes_sum) * 90.0
                calculation = "Monthly total per 90"
            else:
                value = pd.to_numeric(month_df["value"], errors="coerce").mean()
                calculation = "Monthly average"
        else:
            value = raw_values.mean()

        if pd.isna(value):
            continue

        match_count = int(month_df["matchId"].nunique()) if "matchId" in month_df.columns else int(len(month_df))
        minutes_sum = pd.to_numeric(
            month_df.get("minutes_reference", pd.Series(index=month_df.index, dtype=float)),
            errors="coerce",
        ).sum(min_count=1)
        opponents = (
            month_df["oppositionTeamName"]
            .dropna()
            .astype(str)
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )
        opponent_label = ", ".join(opponents[:3])
        if len(opponents) > 3:
            opponent_label = f"{opponent_label} +{len(opponents) - 3}"
        if not opponent_label:
            opponent_label = "Unknown"

        rows.append(
            {
                "matchId": month_start.strftime("%Y-%m"),
                "matchDate": month_start,
                "match_label": month_start.strftime("%b %Y"),
                "oppositionTeamName": opponent_label,
                "metric_key": metric_key,
                "Metric": metric_name,
                "value": float(value),
                "match_count": match_count,
                "minutes": None if pd.isna(minutes_sum) else round(float(minutes_sum), 1),
                "calculation": calculation,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "matchId",
                "matchDate",
                "match_label",
                "oppositionTeamName",
                "metric_key",
                "Metric",
                "value",
                "match_count",
                "minutes",
                "calculation",
            ]
        )

    return pd.DataFrame(rows).sort_values(["matchDate", "Metric"]).reset_index(drop=True)


def _get_period_months(
    chart_df: pd.DataFrame,
    before_period_months: int,
    after_period_months: int,
) -> tuple:
    month_df = (
        chart_df[["matchDate", "match_label"]]
        .drop_duplicates()
        .sort_values("matchDate")
        .reset_index(drop=True)
    )
    if month_df.empty:
        return month_df.copy(), month_df.copy()

    before_count = max(1, min(int(before_period_months), len(month_df)))
    after_count = max(1, min(int(after_period_months), len(month_df)))
    return month_df.head(before_count).copy(), month_df.tail(after_count).copy()


def _format_period_label(period_df: pd.DataFrame) -> str:
    if period_df.empty:
        return "No months"
    labels = period_df["match_label"].astype(str).tolist()
    if len(labels) <= 3:
        return ", ".join(labels)
    return f"{labels[0]} - {labels[-1]} ({len(labels)} months)"


def _render_period_summary(
    *,
    chart_df: pd.DataFrame,
    before_period_months: int,
    after_period_months: int,
    min_minutes_played: int,
) -> None:
    before_df, after_df = _get_period_months(
        chart_df=chart_df,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )
    before_month_keys = set(before_df["matchDate"].astype(str).tolist())
    after_month_keys = set(after_df["matchDate"].astype(str).tolist())
    overlap_count = len(before_month_keys.intersection(after_month_keys))
    month_df = (
        chart_df[["matchDate", "match_label"]]
        .drop_duplicates()
        .sort_values("matchDate")
        .reset_index(drop=True)
    )

    month_chips = []
    for _, row in month_df.iterrows():
        month_key = str(row["matchDate"])
        label = escape(str(row["match_label"]))
        if month_key in before_month_keys and month_key in after_month_keys:
            chip_class = "both"
        elif month_key in before_month_keys:
            chip_class = "before"
        elif month_key in after_month_keys:
            chip_class = "after"
        else:
            chip_class = "neutral"
        month_chips.append(f'<span class="focus-month-chip {chip_class}">{label}</span>')

    before_label = escape(_format_period_label(before_df))
    after_label = escape(_format_period_label(after_df))

    st.markdown(
        f"""
        <style>
            .focus-period-panel {{
                border: 1px solid #E3E7EF;
                border-radius: 8px;
                padding: 12px 14px 14px;
                margin: 8px 0 18px;
                background: #FFFFFF;
            }}
            .focus-period-head {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 16px;
                margin-bottom: 10px;
            }}
            .focus-period-title {{
                font-size: 0.92rem;
                font-weight: 700;
                color: #2F3440;
                margin-bottom: 2px;
            }}
            .focus-period-subtitle {{
                font-size: 0.78rem;
                color: #6B7280;
            }}
            .focus-period-meta {{
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                justify-content: flex-end;
            }}
            .focus-period-pill {{
                border: 1px solid #E3E7EF;
                border-radius: 999px;
                padding: 4px 9px;
                font-size: 0.76rem;
                color: #3F4654;
                background: #F8FAFC;
                white-space: nowrap;
            }}
            .focus-period-rows {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 8px;
                margin-bottom: 10px;
            }}
            .focus-period-row {{
                border-radius: 7px;
                padding: 9px 10px;
                border: 1px solid #E3E7EF;
                min-width: 0;
            }}
            .focus-period-row.before {{
                background: #F4F7FB;
                border-color: #CBD5E0;
            }}
            .focus-period-row.after {{
                background: #FFF8D6;
                border-color: #F2B705;
            }}
            .focus-period-row-label {{
                font-size: 0.75rem;
                font-weight: 700;
                color: #4A5568;
                text-transform: uppercase;
                margin-bottom: 3px;
            }}
            .focus-period-row-months {{
                font-size: 0.88rem;
                color: #2F3440;
                line-height: 1.28;
            }}
            .focus-month-track {{
                display: flex;
                gap: 6px;
                flex-wrap: wrap;
                align-items: center;
            }}
            .focus-month-chip {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-height: 26px;
                border-radius: 999px;
                border: 1px solid #E3E7EF;
                padding: 4px 9px;
                font-size: 0.78rem;
                font-weight: 600;
                color: #667085;
                background: #FFFFFF;
                white-space: nowrap;
            }}
            .focus-month-chip.before {{
                color: #2D3748;
                background: #E9EEF6;
                border-color: #B8C2D2;
            }}
            .focus-month-chip.after {{
                color: #513A00;
                background: #FFECA8;
                border-color: #E6B000;
            }}
            .focus-month-chip.both {{
                color: #2F3440;
                background: linear-gradient(90deg, #E9EEF6 0 50%, #FFECA8 50% 100%);
                border-color: #B8C2D2;
            }}
            .focus-month-chip.neutral {{
                color: #8A93A3;
                background: #FFFFFF;
                border-color: #EEF1F5;
            }}
            @media (max-width: 900px) {{
                .focus-period-head {{
                    display: block;
                }}
                .focus-period-meta {{
                    justify-content: flex-start;
                    margin-top: 8px;
                }}
                .focus-period-rows {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
        <div class="focus-period-panel">
            <div class="focus-period-head">
                <div>
                    <div class="focus-period-title">Comparison windows</div>
                    <div class="focus-period-subtitle">Eligible months after the minutes filter. Colored months are used in the before/after comparison.</div>
                </div>
                <div class="focus-period-meta">
                    <span class="focus-period-pill">Min {int(min_minutes_played)} min</span>
                    <span class="focus-period-pill">Before {len(before_df)}m</span>
                    <span class="focus-period-pill">After {len(after_df)}m</span>
                </div>
            </div>
            <div class="focus-period-rows">
                <div class="focus-period-row before">
                    <div class="focus-period-row-label">Before period</div>
                    <div class="focus-period-row-months">{before_label}</div>
                </div>
                <div class="focus-period-row after">
                    <div class="focus-period-row-label">After period</div>
                    <div class="focus-period-row-months">{after_label}</div>
                </div>
            </div>
            <div class="focus-month-track">
                {''.join(month_chips)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if overlap_count:
        st.caption(
            f"{overlap_count} month(s) appear in both periods because the selected window is larger than the available timeline."
        )


def _add_period_vrects(
    fig: go.Figure,
    *,
    chart_df: pd.DataFrame,
    before_period_months: int,
    after_period_months: int,
) -> None:
    before_df, after_df = _get_period_months(
        chart_df=chart_df,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )

    def add_band(period_df: pd.DataFrame, label: str, color: str) -> None:
        if period_df.empty:
            return
        x0 = pd.to_datetime(period_df["matchDate"].min())
        x1 = pd.to_datetime(period_df["matchDate"].max()) + pd.offsets.MonthEnd(1)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.16,
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
        )

    add_band(before_df, "Before", "#718096")
    add_band(after_df, "After", "#F2B705")


def _build_metric_diagnostics(
    *,
    chart_df: pd.DataFrame,
    selected_metric_keys: List[str],
    metric_labels: Dict[str, str],
    before_period_months: int,
    after_period_months: int,
) -> pd.DataFrame:
    rows = []
    for metric_key in selected_metric_keys:
        metric_df = chart_df[chart_df["metric_key"] == metric_key].sort_values("matchDate")
        values = pd.to_numeric(metric_df["value"], errors="coerce").dropna()
        if values.empty:
            continue

        before_window = max(1, min(int(before_period_months), len(values)))
        after_window = max(1, min(int(after_period_months), len(values)))
        first_avg = float(values.head(before_window).mean())
        last_avg = float(values.tail(after_window).mean())
        latest_value = float(values.iloc[-1])
        delta = last_avg - first_avg
        lower_is_better = _is_lower_better_metric(metric_key, metric_labels)
        improvement = -delta if lower_is_better else delta
        pct_change = np.nan
        if not np.isclose(first_avg, 0):
            pct_change = (delta / abs(first_avg)) * 100.0

        std_value = float(values.std(ddof=0)) if len(values) > 1 else 0.0
        mean_value = float(values.mean())
        consistency = 100.0
        if not np.isclose(mean_value, 0):
            consistency = max(0.0, 100.0 - min(100.0, (std_value / abs(mean_value)) * 100.0))

        slope = 0.0
        if len(values) > 1:
            x_values = np.arange(len(values), dtype=float)
            slope = float(np.polyfit(x_values, values.to_numpy(dtype=float), 1)[0])

        rows.append(
            {
                "metric_key": metric_key,
                "Metric": _format_metric_name(metric_key, metric_labels),
                "Direction": "Lower is better" if lower_is_better else "Higher is better",
                "Months": int(len(values)),
                "Before months": int(before_window),
                "After months": int(after_window),
                "Before avg": round(first_avg, 2),
                "After avg": round(last_avg, 2),
                "Latest": round(latest_value, 2),
                "Change": round(delta, 2),
                "Change %": None if pd.isna(pct_change) else round(pct_change, 1),
                "Improvement": round(improvement, 2),
                "Trend / month": round(slope, 3),
                "Consistency": round(consistency, 1),
            }
        )

    return pd.DataFrame(rows)


def _render_executive_focus_view(
    *,
    chart_df: pd.DataFrame,
    selected_metric_keys: List[str],
    metric_labels: Dict[str, str],
    before_period_months: int,
    after_period_months: int,
) -> None:
    diagnostics_df = _build_metric_diagnostics(
        chart_df=chart_df,
        selected_metric_keys=selected_metric_keys,
        metric_labels=metric_labels,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )
    if diagnostics_df.empty:
        st.info("No diagnostic data available for this focus card.")
        return

    improving_count = int((diagnostics_df["Improvement"] > 0).sum())
    declining_count = int((diagnostics_df["Improvement"] < 0).sum())
    average_consistency = float(diagnostics_df["Consistency"].mean())
    latest_month = chart_df.sort_values("matchDate")["match_label"].iloc[-1]
    before_df, after_df = _get_period_months(
        chart_df=chart_df,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )

    summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
    summary_col_1.metric("KPIs improving", f"{improving_count}/{len(diagnostics_df)}")
    summary_col_2.metric("KPIs declining", str(declining_count))
    summary_col_3.metric("Avg consistency", f"{average_consistency:.0f}/100")
    summary_col_4.metric("Latest month", latest_month)
    st.caption(
        f"Comparison: Before = {_format_period_label(before_df)} | After = {_format_period_label(after_df)}"
    )

    change_df = diagnostics_df.copy()
    change_df["Status"] = np.select(
        [
            change_df["Improvement"] > 0,
            change_df["Improvement"] < 0,
        ],
        ["Improving", "Declining"],
        default="Stable",
    )
    change_df = change_df.sort_values("Improvement", ascending=True)

    fig = px.bar(
        change_df,
        x="Improvement",
        y="Metric",
        orientation="h",
        color="Status",
        color_discrete_map={
            "Improving": "#12A150",
            "Stable": "#7A869A",
            "Declining": "#D64545",
        },
        hover_data={
            "Direction": True,
            "Latest": ":.2f",
            "Change": ":.2f",
            "Consistency": ":.1f",
        },
        title="Focus movement by linked KPI",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#4A5568")
    fig.update_layout(
        xaxis_title="Improvement-adjusted change",
        yaxis_title="",
        legend_title="",
        height=max(320, 78 * len(change_df)),
    )
    st.plotly_chart(fig, use_container_width=True)

    display_cols = [
        "Metric",
        "Direction",
        "Months",
        "Before months",
        "After months",
        "Before avg",
        "After avg",
        "Latest",
        "Change",
        "Change %",
        "Trend / month",
        "Consistency",
    ]
    st.dataframe(diagnostics_df[display_cols], use_container_width=True, hide_index=True)


def _render_timeline_view(
    *,
    chart_df: pd.DataFrame,
    selected_metric_keys: List[str],
    metric_labels: Dict[str, str],
    metric_type_map: Dict[str, str],
    before_period_months: int,
    after_period_months: int,
) -> None:
    fig = go.Figure()
    sorted_df = chart_df.sort_values("matchDate")
    for metric_key in selected_metric_keys:
        metric_df = sorted_df[sorted_df["metric_key"] == metric_key].copy()
        if metric_df.empty:
            continue

        metric_name = _format_metric_name(metric_key, metric_labels)
        metric_df["rolling_value"] = (
            metric_df["value"]
            .rolling(window=min(3, len(metric_df)), min_periods=1)
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=metric_df["matchDate"],
                y=metric_df["value"],
                mode="lines+markers",
                name=metric_name,
                customdata=np.stack(
                    [
                        metric_df["match_label"],
                        metric_df["oppositionTeamName"],
                        metric_df["match_count"],
                        metric_df["calculation"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    + "Matches: %{customdata[2]}<br>"
                    + "Method: %{customdata[3]}<br>"
                    + metric_name
                    + ": %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metric_df["matchDate"],
                y=metric_df["rolling_value"],
                mode="lines",
                name=f"{metric_name} rolling",
                line={"dash": "dash", "width": 2},
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    + metric_name
                    + " rolling: %{y:.2f}<extra></extra>"
                ),
            )
        )

    if selected_metric_keys and all(
        metric_type_map.get(metric) == "percentage"
        for metric in selected_metric_keys
    ):
        fig.update_yaxes(range=[0, 100])

    _add_period_vrects(
        fig,
        chart_df=chart_df,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )
    fig.update_layout(
        title="Monthly evolution with highlighted before/after periods",
        xaxis_title="Month",
        yaxis_title="Metric value",
        legend_title="Metric",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_heatmap_view(
    *,
    scored_df: pd.DataFrame,
) -> None:
    heatmap_df = scored_df.pivot_table(
        index="Metric",
        columns="match_label",
        values="score",
        aggfunc="mean",
    )
    if heatmap_df.empty:
        st.info("No heatmap data available.")
        return

    ordered_matches = (
        scored_df[["match_label", "matchDate"]]
        .drop_duplicates()
        .sort_values("matchDate")["match_label"]
        .tolist()
    )
    heatmap_df = heatmap_df.reindex(columns=ordered_matches)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale=[
                [0.0, "#C93C37"],
                [0.5, "#F2C94C"],
                [1.0, "#1B8A5A"],
            ],
            zmin=0,
            zmax=100,
            colorbar={"title": "Score"},
            hovertemplate="%{y}<br>%{x}<br>Score: %{z:.0f}/100<extra></extra>",
        )
    )
    fig.update_layout(
        title="Traffic-light monthly map",
        xaxis_title="",
        yaxis_title="",
        height=max(320, 90 + 52 * len(heatmap_df.index)),
    )
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)


def _render_before_after_view(
    *,
    scored_df: pd.DataFrame,
    selected_metric_keys: List[str],
    metric_labels: Dict[str, str],
    before_period_months: int,
    after_period_months: int,
) -> None:
    rows = []
    for metric_key in selected_metric_keys:
        metric_df = scored_df[scored_df["metric_key"] == metric_key].sort_values("matchDate")
        values = metric_df["score"].dropna()
        if values.empty:
            continue
        before_window = max(1, min(int(before_period_months), len(values)))
        after_window = max(1, min(int(after_period_months), len(values)))
        rows.append(
            {
                "metric_key": metric_key,
                "Metric": _format_metric_name(metric_key, metric_labels),
                "Before period": float(values.head(before_window).mean()),
                "After period": float(values.tail(after_window).mean()),
            }
        )

    radar_df = pd.DataFrame(rows)
    if radar_df.empty:
        st.info("No before-after data available.")
        return
    before_df, after_df = _get_period_months(
        chart_df=scored_df,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )
    period_title = (
        f"Before: {_format_period_label(before_df)} | "
        f"After: {_format_period_label(after_df)}"
    )

    if len(radar_df) >= 3:
        theta_values = radar_df["Metric"].tolist()
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=radar_df["Before period"],
                theta=theta_values,
                fill="toself",
                name=f"Before ({before_period_months}m)",
                line_color="#A0AEC0",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=radar_df["After period"],
                theta=theta_values,
                fill="toself",
                name=f"After ({after_period_months}m)",
                line_color="#F2B705",
            )
        )
        fig.update_layout(
            title=f"Before vs after profile<br><sup>{period_title}</sup>",
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
            showlegend=True,
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        before_after_long = radar_df.melt(
            id_vars=["Metric"],
            value_vars=["Before period", "After period"],
            var_name="Window",
            value_name="Score",
        )
        fig = px.bar(
            before_after_long,
            x="Metric",
            y="Score",
            color="Window",
            barmode="group",
            range_y=[0, 100],
            title=f"Before vs after score<br><sup>{period_title}</sup>",
            color_discrete_map={
                "Before period": "#A0AEC0",
                "After period": "#F2B705",
            },
        )
        fig.update_layout(xaxis_title="", yaxis_title="Score / 100")
        st.plotly_chart(fig, use_container_width=True)


def _render_month_ranking_view(scored_df: pd.DataFrame) -> None:
    month_score_df = (
        scored_df.groupby(["matchDate", "match_label", "oppositionTeamName"], dropna=False)
        .agg(
            score=("score", "mean"),
            match_count=("match_count", "max"),
            minutes=("minutes", "max"),
            calculation=("calculation", "first"),
        )
        .reset_index()
        .dropna(subset=["score"])
        .sort_values("score", ascending=False)
    )
    if month_score_df.empty:
        st.info("No month ranking data available.")
        return

    month_score_df["Tier"] = pd.cut(
        month_score_df["score"],
        bins=[-0.1, 40, 65, 100],
        labels=["Concern", "Mixed", "Strong"],
    ).astype(str)

    fig = px.bar(
        month_score_df.sort_values("score", ascending=True),
        x="score",
        y="match_label",
        orientation="h",
        color="Tier",
        color_discrete_map={
            "Concern": "#C93C37",
            "Mixed": "#E0A800",
            "Strong": "#1B8A5A",
        },
        hover_data={
            "oppositionTeamName": True,
            "match_count": True,
            "minutes": ":.1f",
            "calculation": True,
            "score": ":.0f",
            "matchDate": "|%b %Y",
        },
        title="Best and weakest months for this focus",
    )
    fig.update_layout(
        xaxis_title="Composite focus score / 100",
        yaxis_title="",
        height=max(360, 44 * len(month_score_df)),
        legend_title="",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_consistency_view(
    *,
    chart_df: pd.DataFrame,
    selected_metric_keys: List[str],
    metric_labels: Dict[str, str],
    before_period_months: int,
    after_period_months: int,
) -> None:
    diagnostics_df = _build_metric_diagnostics(
        chart_df=chart_df,
        selected_metric_keys=selected_metric_keys,
        metric_labels=metric_labels,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )
    if diagnostics_df.empty:
        st.info("No consistency data available.")
        return
    before_df, after_df = _get_period_months(
        chart_df=chart_df,
        before_period_months=before_period_months,
        after_period_months=after_period_months,
    )
    period_title = (
        f"Before: {_format_period_label(before_df)} | "
        f"After: {_format_period_label(after_df)}"
    )

    fig = px.scatter(
        diagnostics_df,
        x="Improvement",
        y="Consistency",
        size="Months",
        color="Direction",
        text="Metric",
        hover_data={
            "Latest": ":.2f",
            "Change": ":.2f",
            "Trend / month": ":.3f",
        },
        title=f"Improvement vs reliability<br><sup>{period_title}</sup>",
        range_y=[0, 105],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#4A5568")
    fig.add_hline(y=70, line_dash="dash", line_color="#A0AEC0")
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Improvement-adjusted change",
        yaxis_title="Consistency / 100",
        legend_title="",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_player_focus_section(
    *,
    player_id: str,
    player_name: str,
    is_admin: bool,
    staff_display_name: str,
    all_kpi_options: List[str],
    selected_kpis: List[str],
    metric_labels: Dict[str, str],
    metric_type_map: Dict[str, str],
    percentage_formula_map: Dict[str, tuple],
    filtered_df: pd.DataFrame,
    event_data: pd.DataFrame,
    attach_minutes_reference: Callable[[pd.DataFrame], pd.DataFrame],
    sheets_client: Any,
    base_dir: str,
) -> None:
    local_file = Path(base_dir) / "data" / "player_focus_cards.csv"
    focus_cards_df = load_focus_cards_df(sheets_client=sheets_client, local_file=local_file)

    player_id_norm = _normalize_player_id(player_id)
    player_name_norm = _clean_text(player_name).lower()

    player_cards_df = focus_cards_df[focus_cards_df["player_id"] == player_id_norm].copy()
    if player_cards_df.empty and player_name_norm:
        player_cards_df = focus_cards_df[
            focus_cards_df["player_name"].str.lower() == player_name_norm
        ].copy()

    if "updated_at" in player_cards_df.columns:
        player_cards_df["updated_at_dt"] = pd.to_datetime(
            player_cards_df["updated_at"],
            errors="coerce",
        )
        player_cards_df = player_cards_df.sort_values(
            "updated_at_dt",
            ascending=False,
            na_position="last",
        ).drop(columns=["updated_at_dt"])

    active_cards_df = player_cards_df[player_cards_df["status"] == "active"].copy()
    strength_df = active_cards_df[active_cards_df["focus_type"] == "Strength"].copy()
    weakness_df = active_cards_df[active_cards_df["focus_type"] == "Weakness"].copy()

    st.info("Player Focus Card: define strengths/weaknesses, link one or more KPIs, and track evolution over time.")
    if is_admin:
        st.caption("Admin mode: you can create, edit, archive and delete cards.")
    else:
        st.caption("Read-only mode: only admins can edit cards.")

    linked_kpis = set()
    for _, row in active_cards_df.iterrows():
        linked_kpis.update(parse_metric_keys(row.get("metric_keys")))

    count_col_1, count_col_2, count_col_3 = st.columns(3)
    count_col_1.metric("Active strengths", int(len(strength_df)))
    count_col_2.metric("Active weaknesses", int(len(weakness_df)))
    count_col_3.metric("Linked KPIs", int(len(linked_kpis)))

    col_strength, col_weakness = st.columns(2)
    with col_strength:
        _render_focus_group(
            title="Strengths",
            cards_df=strength_df,
            metric_labels=metric_labels,
        )
    with col_weakness:
        _render_focus_group(
            title="Weaknesses",
            cards_df=weakness_df,
            metric_labels=metric_labels,
        )

    st.markdown("### Evolution")
    if active_cards_df.empty:
        st.info("No active cards available for this player yet.")
    else:
        option_ids = active_cards_df["card_id"].tolist()
        option_label_map = {
            row["card_id"]: _card_option_label(row)
            for _, row in active_cards_df.iterrows()
        }
        selected_card_id = st.selectbox(
            "Focus card to analyze",
            options=option_ids,
            format_func=lambda card_id: option_label_map.get(card_id, card_id),
            key=f"focus_card_selector_{player_id_norm}",
        )
        selected_card = active_cards_df[
            active_cards_df["card_id"] == selected_card_id
        ].head(1)

        if selected_card.empty:
            st.warning("Selected card is not available.")
        else:
            selected_card_row = selected_card.iloc[0]
            selected_metric_keys = [
                metric_key
                for metric_key in parse_metric_keys(selected_card_row.get("metric_keys"))
                if metric_key in filtered_df.columns
            ]

            if not selected_metric_keys:
                st.warning("This card has no linked KPI available in the current dataset.")
            elif filtered_df.empty:
                st.info("No matches found in the selected filters.")
            else:
                control_col_1, control_col_2, control_col_3, control_col_4 = st.columns(4)
                with control_col_1:
                    min_minutes_played = st.number_input(
                        "Min minutes per match",
                        min_value=0,
                        max_value=120,
                        value=30,
                        step=5,
                        key=f"focus_card_min_minutes_{player_id_norm}",
                    )
                with control_col_2:
                    before_period_months = st.number_input(
                        "Before period (months)",
                        min_value=1,
                        max_value=12,
                        value=1,
                        step=1,
                        key=f"focus_card_before_months_{player_id_norm}",
                    )
                with control_col_3:
                    after_period_months = st.number_input(
                        "After period (months)",
                        min_value=1,
                        max_value=12,
                        value=1,
                        step=1,
                        key=f"focus_card_after_months_{player_id_norm}",
                    )
                with control_col_4:
                    use_per90 = st.toggle(
                        "Use per 90 for non-% KPIs",
                        value=True,
                        key=f"focus_card_use_per90_{player_id_norm}",
                    )

                trends_df = filtered_df.copy()
                trends_df["matchDate"] = pd.to_datetime(trends_df["matchDate"], errors="coerce")
                trends_df = trends_df[trends_df["matchDate"].notna()].sort_values("matchDate")
                if trends_df.empty:
                    st.info("No dated matches available in the selected filters.")
                else:
                    if "oppositionTeamName" not in trends_df.columns:
                        teams_info = (
                            event_data.groupby("matchId")[["oppositionTeamName"]]
                            .first()
                            .reset_index()
                        )
                        trends_df = trends_df.merge(teams_info, on="matchId", how="left")

                    trends_df = attach_minutes_reference(trends_df)
                    trends_df["oppositionTeamName"] = (
                        trends_df.get("oppositionTeamName", pd.Series(index=trends_df.index, dtype="string"))
                        .astype("string")
                        .fillna("Unknown")
                    )

                    minutes_for_filter = pd.to_numeric(
                        trends_df.get("minutes_reference", pd.Series(index=trends_df.index, dtype=float)),
                        errors="coerce",
                    )
                    matches_before_minutes_filter = int(trends_df["matchId"].nunique())
                    if int(min_minutes_played) > 0:
                        trends_df = trends_df[minutes_for_filter >= int(min_minutes_played)].copy()
                    matches_after_minutes_filter = int(trends_df["matchId"].nunique())

                    if trends_df.empty:
                        st.info("No matches remain after applying the minimum minutes filter.")
                    elif matches_before_minutes_filter != matches_after_minutes_filter:
                        removed_matches = matches_before_minutes_filter - matches_after_minutes_filter
                        st.caption(
                            f"{removed_matches} match(es) excluded below {int(min_minutes_played)} minutes."
                        )

                    metric_frames = []
                    for metric_key in (selected_metric_keys if not trends_df.empty else []):
                        raw_values = pd.to_numeric(trends_df[metric_key], errors="coerce")
                        values = raw_values.copy()
                        minutes = pd.to_numeric(
                            trends_df.get("minutes_reference", pd.Series(index=trends_df.index, dtype=float)),
                            errors="coerce",
                        )

                        if use_per90 and metric_type_map.get(metric_key) != "percentage":
                            values = (raw_values / minutes.replace(0, np.nan)) * 90.0

                        metric_df = trends_df[["matchId", "matchDate", "oppositionTeamName"]].copy()
                        metric_df["metric_key"] = metric_key
                        metric_df["Metric"] = _format_metric_name(metric_key, metric_labels)
                        metric_df["value"] = values
                        metric_df["raw_value"] = raw_values
                        metric_df["minutes_reference"] = minutes
                        numerator_col, denominator_col = percentage_formula_map.get(metric_key, (None, None))
                        if numerator_col in trends_df.columns and denominator_col in trends_df.columns:
                            metric_df["numerator_value"] = pd.to_numeric(
                                trends_df[numerator_col],
                                errors="coerce",
                            )
                            metric_df["denominator_value"] = pd.to_numeric(
                                trends_df[denominator_col],
                                errors="coerce",
                            )
                        metric_df = metric_df.dropna(subset=["value"])
                        if not metric_df.empty:
                            metric_frames.append(metric_df)

                    if not metric_frames:
                        st.info("No valid KPI values available for the selected card and filters.")
                    else:
                        match_chart_df = pd.concat(metric_frames, ignore_index=True)
                        chart_df = _build_monthly_focus_df(
                            match_chart_df,
                            metric_type_map=metric_type_map,
                            percentage_formula_map=percentage_formula_map,
                            use_per90=use_per90,
                        )
                        if chart_df.empty:
                            st.info("No monthly KPI values available for the selected card and filters.")
                            return

                        scored_df = _normalize_metric_scores(
                            chart_df=chart_df,
                            metric_labels=metric_labels,
                        )

                        card_title = _clean_text(selected_card_row.get("title")) or "Focus evolution"
                        st.markdown(f"#### {card_title}")
                        focus_type = _clean_text(selected_card_row.get("focus_type"))
                        if focus_type:
                            month_count = chart_df["matchDate"].nunique()
                            match_count = match_chart_df["matchId"].nunique()
                            st.caption(
                                f"{focus_type} monthly analysis across {month_count} month(s) and {match_count} match(es)."
                            )
                        _render_period_summary(
                            chart_df=chart_df,
                            before_period_months=int(before_period_months),
                            after_period_months=int(after_period_months),
                            min_minutes_played=int(min_minutes_played),
                        )

                        (
                            executive_tab,
                            timeline_tab,
                            heatmap_tab,
                            profile_tab,
                            match_tab,
                            consistency_tab,
                        ) = st.tabs(
                            [
                                "Executive",
                                "Timeline",
                                "Heatmap",
                                "Before / Now",
                                "Month ranking",
                                "Consistency",
                            ]
                        )

                        with executive_tab:
                            _render_executive_focus_view(
                                chart_df=chart_df,
                                selected_metric_keys=selected_metric_keys,
                                metric_labels=metric_labels,
                                before_period_months=int(before_period_months),
                                after_period_months=int(after_period_months),
                            )

                        with timeline_tab:
                            _render_timeline_view(
                                chart_df=chart_df,
                                selected_metric_keys=selected_metric_keys,
                                metric_labels=metric_labels,
                                metric_type_map=metric_type_map,
                                before_period_months=int(before_period_months),
                                after_period_months=int(after_period_months),
                            )

                        with heatmap_tab:
                            _render_heatmap_view(scored_df=scored_df)

                        with profile_tab:
                            _render_before_after_view(
                                scored_df=scored_df,
                                selected_metric_keys=selected_metric_keys,
                                metric_labels=metric_labels,
                                before_period_months=int(before_period_months),
                                after_period_months=int(after_period_months),
                            )

                        with match_tab:
                            _render_month_ranking_view(scored_df=scored_df)

                        with consistency_tab:
                            _render_consistency_view(
                                chart_df=chart_df,
                                selected_metric_keys=selected_metric_keys,
                                metric_labels=metric_labels,
                                before_period_months=int(before_period_months),
                                after_period_months=int(after_period_months),
                            )

    st.markdown("---")
    if is_admin:
        st.markdown("### Admin Controls")

        default_metrics = [k for k in (selected_kpis or all_kpi_options) if k in all_kpi_options][:2]
        with st.form(f"focus_card_create_form_{player_id_norm}", clear_on_submit=True):
            st.markdown("#### Create focus card")
            new_focus_type = st.selectbox(
                "Type",
                options=["Strength", "Weakness"],
                key=f"focus_card_create_type_{player_id_norm}",
            )
            new_title = st.text_input(
                "Title",
                placeholder="e.g. Ball progression under pressure",
                key=f"focus_card_create_title_{player_id_norm}",
            )
            new_description = st.text_area(
                "Description",
                placeholder="Context, coaching cue, expected behavior...",
                key=f"focus_card_create_description_{player_id_norm}",
            )
            new_metric_keys = st.multiselect(
                "Linked KPIs",
                options=all_kpi_options,
                default=default_metrics,
                format_func=lambda metric: _format_metric_name(metric, metric_labels),
                key=f"focus_card_create_metrics_{player_id_norm}",
            )
            create_clicked = st.form_submit_button("Create card")

        if create_clicked:
            if not _clean_text(new_title):
                st.warning("Title is required.")
            elif not new_metric_keys:
                st.warning("Select at least one KPI.")
            else:
                now_ts = _now_timestamp()
                new_row = {
                    "card_id": uuid.uuid4().hex,
                    "player_id": player_id_norm,
                    "player_name": _clean_text(player_name),
                    "focus_type": _normalize_focus_type(new_focus_type),
                    "title": _clean_text(new_title),
                    "description": _clean_text(new_description),
                    "metric_keys": _serialize_metric_keys(new_metric_keys),
                    "status": "active",
                    "created_at": now_ts,
                    "updated_at": now_ts,
                    "created_by": _clean_text(staff_display_name) or "Admin",
                    "updated_by": _clean_text(staff_display_name) or "Admin",
                }
                updated_df = pd.concat([focus_cards_df, pd.DataFrame([new_row])], ignore_index=True)
                destination = save_focus_cards_df(
                    df=updated_df,
                    sheets_client=sheets_client,
                    local_file=local_file,
                )
                st.success(f"Card created. Saved to {destination}.")
                st.rerun()

        if not player_cards_df.empty:
            st.markdown("#### Edit focus card")
            editable_option_ids = player_cards_df["card_id"].tolist()
            editable_label_map = {
                row["card_id"]: _card_option_label(row)
                for _, row in player_cards_df.iterrows()
            }
            edit_card_id = st.selectbox(
                "Card",
                options=editable_option_ids,
                format_func=lambda card_id: editable_label_map.get(card_id, card_id),
                key=f"focus_card_edit_selector_{player_id_norm}",
            )
            edit_row_df = player_cards_df[player_cards_df["card_id"] == edit_card_id].head(1)

            if not edit_row_df.empty:
                edit_row = edit_row_df.iloc[0]
                edit_metrics_default = [
                    metric for metric in parse_metric_keys(edit_row.get("metric_keys"))
                    if metric in all_kpi_options
                ]

                with st.form(f"focus_card_edit_form_{player_id_norm}"):
                    edit_focus_type = st.selectbox(
                        "Type",
                        options=["Strength", "Weakness"],
                        index=0 if _normalize_focus_type(edit_row.get("focus_type")) == "Strength" else 1,
                    )
                    edit_title = st.text_input("Title", value=_clean_text(edit_row.get("title")))
                    edit_description = st.text_area(
                        "Description",
                        value=_clean_text(edit_row.get("description")),
                    )
                    edit_metric_keys = st.multiselect(
                        "Linked KPIs",
                        options=all_kpi_options,
                        default=edit_metrics_default,
                        format_func=lambda metric: _format_metric_name(metric, metric_labels),
                    )
                    edit_status = st.selectbox(
                        "Status",
                        options=["active", "archived"],
                        index=0 if _normalize_focus_status(edit_row.get("status")) == "active" else 1,
                    )
                    save_edit_clicked = st.form_submit_button("Save changes")

                if save_edit_clicked:
                    if not _clean_text(edit_title):
                        st.warning("Title is required.")
                    elif not edit_metric_keys:
                        st.warning("Select at least one KPI.")
                    else:
                        updated_df = focus_cards_df.copy()
                        mask = updated_df["card_id"] == edit_card_id
                        if mask.any():
                            updated_df.loc[mask, "player_id"] = player_id_norm
                            updated_df.loc[mask, "player_name"] = _clean_text(player_name)
                            updated_df.loc[mask, "focus_type"] = _normalize_focus_type(edit_focus_type)
                            updated_df.loc[mask, "title"] = _clean_text(edit_title)
                            updated_df.loc[mask, "description"] = _clean_text(edit_description)
                            updated_df.loc[mask, "metric_keys"] = _serialize_metric_keys(edit_metric_keys)
                            updated_df.loc[mask, "status"] = _normalize_focus_status(edit_status)
                            updated_df.loc[mask, "updated_at"] = _now_timestamp()
                            updated_df.loc[mask, "updated_by"] = _clean_text(staff_display_name) or "Admin"
                            destination = save_focus_cards_df(
                                df=updated_df,
                                sheets_client=sheets_client,
                                local_file=local_file,
                            )
                            st.success(f"Card updated. Saved to {destination}.")
                            st.rerun()

                delete_state_key = f"focus_card_delete_confirm_{player_id_norm}_{edit_card_id}"
                if st.session_state.get(delete_state_key, False):
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Confirm delete",
                            type="primary",
                            key=f"focus_card_confirm_delete_btn_{player_id_norm}_{edit_card_id}",
                        ):
                            updated_df = focus_cards_df[focus_cards_df["card_id"] != edit_card_id].copy()
                            destination = save_focus_cards_df(
                                df=updated_df,
                                sheets_client=sheets_client,
                                local_file=local_file,
                            )
                            st.session_state.pop(delete_state_key, None)
                            st.success(f"Card deleted. Saved to {destination}.")
                            st.rerun()
                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"focus_card_cancel_delete_btn_{player_id_norm}_{edit_card_id}",
                        ):
                            st.session_state.pop(delete_state_key, None)
                            st.rerun()
                else:
                    if st.button(
                        "Delete selected card",
                        key=f"focus_card_delete_btn_{player_id_norm}_{edit_card_id}",
                    ):
                        st.session_state[delete_state_key] = True
                        st.warning("Press 'Confirm delete' to permanently remove this card.")
    else:
        st.caption("Only admin users can manage focus cards.")
