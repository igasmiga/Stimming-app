from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "scripts"))
from imu_pipeline import add_motion_features, detect_motion_episodes, fuse_sensor_features, read_ximu_csv, session_summary


st.set_page_config(page_title="Stimalyzer – analiza ruchu", layout="wide")
ANNOTATION_COLUMNS = ["session_id", "start_s", "end_s", "label", "annotator", "comment"]
LABELS = [
    "neutral",
    "x_flapping",
    "hand_flapping",
    "rocking_side",
    "rocking_forward_back",
    "walking",
    "toe_walking",
    "spinning",
    "clapping",
    "other_repetitive_movement",
]


def seconds_as_clock(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def add_neutral_gaps(annotations: pd.DataFrame, session_id: str, duration_s: float, annotator: str) -> pd.DataFrame:
    """Dodaje neutral tylko do nieopisanych fragmentów po obejrzeniu sesji."""
    rows = annotations.sort_values("start_s").to_dict("records")
    cursor = 0.0
    gaps: list[dict] = []
    for row in rows:
        start = max(0.0, float(row["start_s"]))
        end = min(duration_s, float(row["end_s"]))
        if start > cursor:
            gaps.append({"session_id": session_id, "start_s": round(cursor, 2), "end_s": round(start, 2), "label": "neutral", "annotator": annotator, "comment": "uzupełnione automatycznie"})
        cursor = max(cursor, end)
    if cursor < duration_s:
        gaps.append({"session_id": session_id, "start_s": round(cursor, 2), "end_s": round(duration_s, 2), "label": "neutral", "annotator": annotator, "comment": "uzupełnione automatycznie"})
    return pd.concat([annotations, pd.DataFrame(gaps)], ignore_index=True).sort_values("start_s").reset_index(drop=True)


def annotation_problems(annotations: pd.DataFrame) -> list[str]:
    ordered = annotations.sort_values("start_s")
    problems: list[str] = []
    rows = list(ordered.itertuples())
    for row in rows:
        if row.end_s <= row.start_s:
            problems.append(f"Niepoprawny przedział: {row.start_s}–{row.end_s} s.")
    for previous, current in zip(rows, rows[1:]):
        if current.start_s < previous.end_s:
            problems.append(f"Nakładające się etykiety: {previous.start_s}–{previous.end_s} s oraz {current.start_s}–{current.end_s} s.")
    return problems


@st.cache_data(show_spinner=False)
def analyse_session(
    wrist_bytes: bytes,
    lumbar_bytes: bytes,
    threshold: float,
    minimum_duration: float,
    wrist_sync_s: float,
    lumbar_sync_s: float,
):
    wrist = read_ximu_csv(wrist_bytes)
    lumbar = read_ximu_csv(lumbar_bytes)
    wrist_features = add_motion_features(wrist)
    lumbar_features = add_motion_features(lumbar)
    wrist_scored, _ = detect_motion_episodes(
        wrist_features, threshold=threshold, min_duration_seconds=minimum_duration
    )
    lumbar_scored, _ = detect_motion_episodes(
        lumbar_features, threshold=threshold, min_duration_seconds=minimum_duration
    )
    lumbar_offset = wrist_sync_s - lumbar_sync_s
    featured = fuse_sensor_features(wrist, lumbar, lumbar_time_offset_s=lumbar_offset)
    analysed, episodes = detect_motion_episodes(
        featured, threshold=threshold, min_duration_seconds=minimum_duration
    )
    shared_start = float(featured.attrs["shared_start_s"])
    wrist_scored = wrist_scored.copy()
    lumbar_scored = lumbar_scored.copy()
    wrist_scored["time_s"] -= shared_start
    lumbar_scored["time_s"] += lumbar_offset - shared_start
    wrist_scored = wrist_scored[wrist_scored["time_s"] >= 0].reset_index(drop=True)
    lumbar_scored = lumbar_scored[lumbar_scored["time_s"] >= 0].reset_index(drop=True)
    return analysed, episodes, session_summary(analysed, episodes), wrist_scored, lumbar_scored


st.title("Stimalyzer")
st.caption("Prototyp do ilościowej analizy ruchu z czujników x-IMU3 — nie jest narzędziem diagnostycznym.")

with st.sidebar:
    st.header("Parametry detektora")
    threshold = st.slider("Czułość detekcji", 1.5, 8.0, 3.5, 0.5)
    minimum_duration = st.slider("Minimalny czas epizodu (s)", 0.5, 10.0, 1.0, 0.5)
    st.caption("Niższy próg wykrywa więcej ruchów. Każdy epizod trzeba samodzielnie sprawdzić.")

with st.form("session"):
    left, right = st.columns(2)
    with left:
        participant_id = st.text_input("Pseudonim / ID uczestnika", placeholder="np. P-001")
        session_id = st.text_input("ID sesji", value="SES-001")
    with right:
        session_date = st.date_input("Data sesji")
        st.text_input("Czujniki", value="nadgarstek + odcinek lędźwiowy", disabled=True)
    notes = st.text_area("Notatki z obserwacji", placeholder="Np. 02:15–02:25: machanie rękami")
    wrist_file = st.file_uploader("Plik Inertial.csv — nadgarstek", type=["csv"], key="wrist")
    lumbar_file = st.file_uploader("Plik Inertial.csv — odcinek lędźwiowy", type=["csv"], key="lumbar")
    st.markdown("**Synchronizacja:** znajdź ten sam skok/ruch na obu wykresach i wpisz czas jego wystąpienia w każdym pliku. Jeśli nie znasz jeszcze czasu, wpisz tymczasowo `0` dla obu.")
    sync_left, sync_right = st.columns(2)
    with sync_left:
        wrist_sync_s = st.number_input("Czas skoku — nadgarstek [s]", min_value=0.0, value=0.0, step=0.1)
    with sync_right:
        lumbar_sync_s = st.number_input("Czas skoku — lędźwie [s]", min_value=0.0, value=0.0, step=0.1)
    submitted = st.form_submit_button("Analizuj sesję", type="primary", use_container_width=True)

if submitted:
    if not wrist_file or not lumbar_file or not participant_id.strip():
        st.error("Podaj pseudonim uczestnika i wgraj oba pliki Inertial.csv.")
    else:
        try:
            with st.spinner("Przetwarzanie sygnałów IMU..."):
                frame, episodes, summary, wrist_frame, lumbar_frame = analyse_session(
                    wrist_file.getvalue(), lumbar_file.getvalue(), threshold, minimum_duration,
                    wrist_sync_s, lumbar_sync_s,
                )
            st.session_state.result = (frame, episodes, summary, wrist_frame, lumbar_frame)
            st.session_state.metadata = {
                "participant_id": participant_id.strip(), "session_id": session_id.strip(),
                "session_date": str(session_date), "notes": notes,
                "wrist_sync_s": wrist_sync_s, "lumbar_sync_s": lumbar_sync_s,
            }
            st.session_state.annotations = empty_annotations()
            st.session_state.annotation_revision = st.session_state.get("annotation_revision", 0) + 1
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))

if "result" in st.session_state:
    frame, episodes, summary, wrist_frame, lumbar_frame = st.session_state.result
    metadata = st.session_state.metadata
    st.divider()
    st.subheader(f"Sesja {metadata['session_id']} — {metadata['participant_id']}")
    st.caption(f"Czujniki: nadgarstek + odcinek lędźwiowy · data: {metadata['session_date']}")
    st.info(
        "Wspólny czas 0 s został ustawiony na pierwszy moment dostępny w obu czujnikach. "
        f"Synchronizacja: skok nadgarstek {metadata['wrist_sync_s']:.1f} s, "
        f"lędźwie {metadata['lumbar_sync_s']:.1f} s. "
        "Wszystkie etykiety zapisujesz już w tej wspólnej osi czasu."
    )

    columns = st.columns(5)
    columns[0].metric("Czas sesji", seconds_as_clock(summary["duration_s"]))
    columns[1].metric("Próbkowanie", f"{summary['sampling_rate_hz']} Hz")
    columns[2].metric("Próbki", f"{summary['samples']:,}")
    columns[3].metric("Kandydaci na epizody", summary["episode_count"])
    columns[4].metric("Czas kandydatów", f"{summary['candidate_duration_s']} s")

    inspect_range = st.slider(
        "Fragment sesji do oglądania i etykietowania [s]",
        min_value=0.0,
        max_value=float(summary["duration_s"]),
        value=(0.0, min(60.0, float(summary["duration_s"]))),
        step=0.5,
        help="Wybierz krótki fragment, aby wykresy były czytelne. Ten zakres możesz od razu zapisać jako etykietę.",
    )
    visible_start, visible_end = inspect_range
    plot_frame = frame[(frame["time_s"] >= visible_start) & (frame["time_s"] <= visible_end)]
    plot_frame = plot_frame.iloc[:: max(1, len(plot_frame) // 5000)].copy()
    figure = px.line(
        plot_frame, x="time_s", y="motion_score",
        labels={"time_s": "Czas od początku sesji [s]", "motion_score": "Dynamika ruchu"},
        title="Połączona dynamika ruchu nadgarstka i odcinka lędźwiowego",
    )
    for episode in episodes.itertuples():
        figure.add_vrect(x0=episode.start_s, x1=episode.end_s, fillcolor="#ff9f43", opacity=0.18, line_width=0)
    st.plotly_chart(figure, use_container_width=True)
    st.caption("Najedź kursorem na wykres, aby odczytać czas w sekundach. Pomarańczowe pola są tylko podpowiedziami detektora.")

    st.subheader("Sześć wykresów sygnałów IMU")
    st.caption("Dla każdego czujnika widzisz: akcelerometr X/Y/Z, żyroskop X/Y/Z oraz uproszczoną dynamikę ruchu. Najedź kursorem, aby odczytać czas i wartość.")

    sensor_specs = [
        ("Nadgarstek", wrist_frame),
        ("Odcinek lędźwiowy", lumbar_frame),
    ]
    chart_specs = [
        (
            "Akcelerometr — osie X, Y, Z",
            ["Accelerometer X (g)", "Accelerometer Y (g)", "Accelerometer Z (g)"],
            "Przyspieszenie [g]",
        ),
        (
            "Żyroskop — osie X, Y, Z",
            ["Gyroscope X (deg/s)", "Gyroscope Y (deg/s)", "Gyroscope Z (deg/s)"],
            "Prędkość kątowa [°/s]",
        ),
        (
            "Dynamika ruchu",
            ["motion_score"],
            "Wynik dynamiki",
        ),
    ]
    line_colors = ["#e45756", "#54a24b", "#4c78a8"]

    for sensor_name, sensor_frame in sensor_specs:
        st.markdown(f"#### {sensor_name}")
        sensor_plot = sensor_frame[(sensor_frame["time_s"] >= visible_start) & (sensor_frame["time_s"] <= visible_end)]
        sensor_plot = sensor_plot.iloc[:: max(1, len(sensor_plot) // 5000)]
        for chart_title, y_columns, y_label in chart_specs:
            figure = px.line(
                sensor_plot,
                x="time_s",
                y=y_columns,
                labels={
                    "time_s": "Czas od początku sesji [s]",
                    "value": y_label,
                    "variable": "Oś / cecha",
                },
                title=f"{sensor_name}: {chart_title}",
                color_discrete_sequence=line_colors,
            )
            figure.update_layout(hovermode="x unified", height=300, legend_title_text="")
            st.plotly_chart(figure, use_container_width=True)

    st.subheader("Dodaj etykietę ruchu")
    suggestion_options = {
        "Aktualnie oglądany fragment": (visible_start, visible_end),
        "Własny przedział czasu": (0.0, min(1.0, summary["duration_s"])),
    }
    suggestion_options.update({
        f"Epizod {row.episode_id}: {seconds_as_clock(row.start_s)}–{seconds_as_clock(row.end_s)}": (row.start_s, row.end_s)
        for row in episodes.itertuples()
    })
    selected_suggestion = st.selectbox("Użyj podpowiedzi detektora albo wpisz własny czas", list(suggestion_options))
    default_start, default_end = suggestion_options[selected_suggestion]

    with st.form("add_annotation", clear_on_submit=True):
        first, second, third = st.columns(3)
        with first:
            start_s = st.number_input("Początek [s]", min_value=0.0, max_value=float(summary["duration_s"]), value=float(default_start), step=0.1)
        with second:
            end_s = st.number_input("Koniec [s]", min_value=0.0, max_value=float(summary["duration_s"]), value=float(default_end), step=0.1)
        with third:
            label = st.selectbox("Rodzaj ruchu", LABELS)
        annotator = st.text_input("Twoje inicjały", value="IM")
        comment = st.text_input("Notatka (opcjonalnie)", placeholder="Np. pewne oznaczenie z wideo")
        add_annotation = st.form_submit_button("Dodaj etykietę")

    if add_annotation:
        if end_s <= start_s:
            st.error("Koniec musi być później niż początek.")
        elif not annotator.strip():
            st.error("Wpisz inicjały osoby oznaczającej.")
        else:
            new_row = pd.DataFrame([{
                "session_id": metadata["session_id"], "start_s": round(start_s, 2),
                "end_s": round(end_s, 2), "label": label,
                "annotator": annotator.strip(), "comment": comment.strip(),
            }])
            st.session_state.annotations = pd.concat(
                [st.session_state.annotations, new_row], ignore_index=True
            ).sort_values("start_s").reset_index(drop=True)
            st.session_state.annotation_revision += 1
            st.rerun()

    st.subheader("Twoje etykiety")
    edited_annotations = st.data_editor(
        st.session_state.annotations,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "start_s": st.column_config.NumberColumn("Początek [s]", min_value=0.0, max_value=float(summary["duration_s"]), step=0.1),
            "end_s": st.column_config.NumberColumn("Koniec [s]", min_value=0.0, max_value=float(summary["duration_s"]), step=0.1),
            "label": st.column_config.SelectboxColumn("Rodzaj ruchu", options=LABELS),
        },
        key=f"annotation_editor_{st.session_state.annotation_revision}",
    )
    st.session_state.annotations = edited_annotations[ANNOTATION_COLUMNS]
    problems = annotation_problems(st.session_state.annotations)
    if problems:
        st.error("Popraw etykiety przed pobraniem pliku: " + " ".join(problems))
    elif not st.session_state.annotations.empty:
        labelled_duration = (st.session_state.annotations["end_s"] - st.session_state.annotations["start_s"]).sum()
        st.success(f"Etykiety są poprawne. Opisany czas: {labelled_duration:.1f} s z {summary['duration_s']:.1f} s.")

    with st.expander("Uzupełnij nieopisane fragmenty jako neutral"):
        st.warning("Użyj tego tylko po obejrzeniu całej sesji. Każda luka między etykietami zostanie zapisana jako neutral.")
        neutral_annotator = st.text_input("Inicjały do etykiet neutral", value="IM", key="neutral_annotator")
        if st.button("Dodaj neutral do wszystkich luk"):
            if not neutral_annotator.strip():
                st.error("Wpisz inicjały.")
            elif annotation_problems(st.session_state.annotations):
                st.error("Najpierw popraw nakładające się lub błędne etykiety.")
            else:
                st.session_state.annotations = add_neutral_gaps(
                    st.session_state.annotations,
                    metadata["session_id"],
                    float(summary["duration_s"]),
                    neutral_annotator.strip(),
                )
                st.session_state.annotation_revision += 1
                st.rerun()
    st.download_button(
        "Pobierz gotowy plik annotations.csv dla tej sesji",
        st.session_state.annotations.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"annotations_{metadata['session_id']}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if metadata["notes"]:
        st.info(f"Notatki obserwatora: {metadata['notes']}")
    st.warning("Wyniki przedstawiają kandydatów na dynamiczne ruchy. Nie są rozpoznaniem stymowania ani podstawą diagnozy ASD.")
