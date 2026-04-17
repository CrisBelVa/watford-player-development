import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
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

                    metric_frames = []
                    for metric_key in selected_metric_keys:
                        raw_values = pd.to_numeric(trends_df[metric_key], errors="coerce")
                        values = raw_values.copy()

                        if use_per90 and metric_type_map.get(metric_key) != "percentage":
                            minutes = pd.to_numeric(
                                trends_df.get("minutes_reference", pd.Series(index=trends_df.index, dtype=float)),
                                errors="coerce",
                            ).replace(0, np.nan)
                            values = (raw_values / minutes) * 90.0

                        metric_df = trends_df[["matchId", "matchDate", "oppositionTeamName"]].copy()
                        metric_df["metric_key"] = metric_key
                        metric_df["Metric"] = _format_metric_name(metric_key, metric_labels)
                        metric_df["value"] = values
                        metric_df = metric_df.dropna(subset=["value"])
                        if not metric_df.empty:
                            metric_frames.append(metric_df)

                    if not metric_frames:
                        st.info("No valid KPI values available for the selected card and filters.")
                    else:
                        chart_df = pd.concat(metric_frames, ignore_index=True)
                        chart_df["match_label"] = (
                            chart_df["matchDate"].dt.strftime("%Y-%m-%d")
                            + " vs "
                            + chart_df["oppositionTeamName"].astype(str)
                        )

                        fig = px.line(
                            chart_df.sort_values("matchDate"),
                            x="matchDate",
                            y="value",
                            color="Metric",
                            markers=True,
                            hover_data={
                                "match_label": True,
                                "Metric": True,
                                "value": ":.2f",
                                "matchDate": False,
                            },
                            title=_clean_text(selected_card_row.get("title")) or "Focus evolution",
                        )
                        if selected_metric_keys and all(
                            metric_type_map.get(metric) == "percentage"
                            for metric in selected_metric_keys
                        ):
                            fig.update_yaxes(range=[0, 100])

                        fig.update_layout(
                            xaxis_title="Match date",
                            yaxis_title="Metric value",
                            legend_title="Metric",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        progress_df = _build_progress_rows(
                            chart_df=chart_df,
                            selected_metric_keys=selected_metric_keys,
                            metric_labels=metric_labels,
                        )
                        if not progress_df.empty:
                            st.caption("Change compares average of first matches vs latest matches in current filter window.")
                            st.dataframe(progress_df, use_container_width=True, hide_index=True)

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
