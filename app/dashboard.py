"""Dart Tracker analytics dashboard."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from app.database import get_connection, init_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dart Tracker", page_icon="🎯", layout="wide")

PLAYER_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
BOARD_DARK    = "#131313"
BOARD_LIGHT   = "#E0D4AF"
RING_RED      = "#A0192B"
RING_GREEN    = "#186526"
ZONE_COLORS = {
    "triple":   "#1baf7a",
    "double":   "#2a78d6",
    "single":   "#eda100",
    "bull":     "#4a3aa7",
    "bullseye": "#9b6ee8",
    "miss":     "#e34948",
}
TRANSPARENT_BG = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
GRID_COLOR = "rgba(128,128,128,0.15)"
SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
NO_MODEBAR = {"displayModeBar": False}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def load_players():
    with get_connection() as conn:
        rows = conn.execute("SELECT id, name FROM players ORDER BY name").fetchall()
    return _df(rows)


def load_throws(player_id: int, game_type: str | None = None) -> pd.DataFrame:
    q = """
        SELECT t.*, s.game_type, s.started_at
        FROM throws t
        JOIN sessions s ON t.session_id = s.id
        WHERE t.player_id = ?
    """
    params = [player_id]
    if game_type:
        q += " AND s.game_type = ?"
        params.append(game_type)
    q += " ORDER BY t.id"
    with get_connection() as conn:
        rows = conn.execute(q, params).fetchall()
    return _df(rows)


def load_session_stats(player_id: int, game_type: str | None = None) -> pd.DataFrame:
    q = """
        SELECT
            s.id AS session_id,
            s.game_type,
            s.started_at,
            COUNT(t.id)   AS throw_count,
            COALESCE(SUM(t.score_value), 0) AS total_score,
            CASE WHEN COUNT(t.id) > 0
                 THEN ROUND(SUM(t.score_value) * 3.0 / COUNT(t.id), 2)
                 ELSE 0
            END AS three_dart_avg
        FROM sessions s
        JOIN session_players sp ON s.id = sp.session_id AND sp.player_id = ?
        LEFT JOIN throws t      ON t.session_id = s.id AND t.player_id = ?
    """
    params = [player_id, player_id]
    if game_type:
        q += " AND s.game_type = ?"
        params.append(game_type)
    q += " GROUP BY s.id ORDER BY s.started_at"
    with get_connection() as conn:
        rows = conn.execute(q, params).fetchall()
    return _df(rows)


def load_all_players_session_stats() -> pd.DataFrame:
    """Last 12 sessions per player for comparison chart."""
    q = """
        SELECT
            p.id AS player_id, p.name,
            s.id AS session_id, s.started_at,
            CASE WHEN COUNT(t.id) > 0
                 THEN ROUND(SUM(t.score_value) * 3.0 / COUNT(t.id), 2)
                 ELSE 0
            END AS three_dart_avg
        FROM players p
        JOIN session_players sp ON p.id = sp.player_id
        JOIN sessions s         ON s.id = sp.session_id
        LEFT JOIN throws t      ON t.session_id = s.id AND t.player_id = p.id
        GROUP BY p.id, s.id
        ORDER BY p.name, s.started_at
    """
    with get_connection() as conn:
        rows = conn.execute(q).fetchall()
    df = _df(rows)
    if df.empty:
        return df
    # Keep last 12 sessions per player (pandas-3-safe)
    df = (df.sort_values(["player_id", "started_at"])
            .groupby("player_id", group_keys=False)
            .tail(12)
            .reset_index(drop=True))
    return df

# ---------------------------------------------------------------------------
# Dartboard drawing
# ---------------------------------------------------------------------------

def _segment_angle(idx: int) -> float:
    """Centre angle (degrees clockwise from top) of segment at index idx."""
    return idx * 18.0


def build_dartboard_fig(throws_df: pd.DataFrame, mode: str = "scatter") -> go.Figure:
    fig = go.Figure()

    # Radii (mm): bullseye=12, bull=29, triple_inner=93, triple_outer=102,
    #             double_inner=147, double_outer=157
    radii = {
        "bull_inner": 12, "bull_outer": 29,
        "triple_inner": 93, "triple_outer": 102,
        "double_inner": 147, "double_outer": 157,
    }
    board_r = 170

    def polar_path(r_inner, r_outer, ang_start, ang_end, n=30):
        angles = [ang_start + (ang_end - ang_start) * i / n for i in range(n + 1)]
        xs, ys = [], []
        for a in angles:
            rad = math.radians(a - 90)
            xs.append(r_outer * math.cos(rad))
            ys.append(r_outer * math.sin(rad))
        for a in reversed(angles):
            rad = math.radians(a - 90)
            xs.append(r_inner * math.cos(rad))
            ys.append(r_inner * math.sin(rad))
        xs.append(xs[0]); ys.append(ys[0])
        return xs, ys

    # --- Draw 20 segments (single areas, double ring, triple ring) ---
    # Build hit-frequency map for heat mode
    heat_map: dict[int, int] = {}
    if mode == "heat" and not throws_df.empty:
        for seg in SEGMENT_ORDER:
            heat_map[seg] = int((throws_df["segment"] == seg).sum())
        max_hits = max(heat_map.values()) if heat_map else 1

    for i, seg in enumerate(SEGMENT_ORDER):
        ang_start = i * 18 - 9
        ang_end   = i * 18 + 9
        is_dark = (i % 2 == 0)
        base_color = BOARD_DARK if is_dark else BOARD_LIGHT

        alpha = 1.0
        if mode == "heat" and heat_map:
            alpha = 0.25 + 0.75 * (heat_map.get(seg, 0) / max(max_hits, 1))

        def _rgba(hex_color, a=1.0):
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{a:.2f})"

        # Single area (outer bull to inner triple)
        xs, ys = polar_path(radii["bull_outer"], radii["triple_inner"], ang_start, ang_end)
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                                 fillcolor=_rgba(base_color, alpha),
                                 line=dict(color="rgba(60,60,60,0.4)", width=0.5),
                                 mode="lines", hoverinfo="skip", showlegend=False))

        # Single area (triple_outer to double_inner)
        xs, ys = polar_path(radii["triple_outer"], radii["double_inner"], ang_start, ang_end)
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                                 fillcolor=_rgba(base_color, alpha),
                                 line=dict(color="rgba(60,60,60,0.4)", width=0.5),
                                 mode="lines", hoverinfo="skip", showlegend=False))

        # Triple ring
        xs, ys = polar_path(radii["triple_inner"], radii["triple_outer"], ang_start, ang_end)
        ring_color = RING_RED if is_dark else RING_GREEN
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                                 fillcolor=_rgba(ring_color, alpha),
                                 line=dict(color="rgba(60,60,60,0.4)", width=0.5),
                                 mode="lines", hoverinfo="skip", showlegend=False))

        # Double ring
        xs, ys = polar_path(radii["double_inner"], radii["double_outer"], ang_start, ang_end)
        fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                                 fillcolor=_rgba(ring_color, alpha),
                                 line=dict(color="rgba(60,60,60,0.4)", width=0.5),
                                 mode="lines", hoverinfo="skip", showlegend=False))

        # Segment label
        label_r = radii["double_outer"] + 10
        ang_mid = i * 18
        lx = label_r * math.cos(math.radians(ang_mid - 90))
        ly = label_r * math.sin(math.radians(ang_mid - 90))
        fig.add_annotation(x=lx, y=ly, text=str(seg),
                           font=dict(size=9, color="rgba(80,80,80,0.9)"), showarrow=False)

    # Bull rings
    theta = list(range(361))
    def circle_xy(r):
        xs = [r * math.cos(math.radians(a - 90)) for a in theta]
        ys = [r * math.sin(math.radians(a - 90)) for a in theta]
        return xs, ys

    xs, ys = circle_xy(radii["bull_outer"])
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                             fillcolor=RING_GREEN, line=dict(color="#333", width=1),
                             mode="lines", hoverinfo="skip", showlegend=False))
    xs, ys = circle_xy(radii["bull_inner"])
    fig.add_trace(go.Scatter(x=xs, y=ys, fill="toself",
                             fillcolor=RING_RED, line=dict(color="#333", width=1),
                             mode="lines", hoverinfo="skip", showlegend=False))

    # --- Throws overlay ---
    if not throws_df.empty:
        if mode == "scatter":
            valid = throws_df.dropna(subset=["x", "y"])
            ring_col = valid["ring"].map(lambda r: ZONE_COLORS.get(r, "#999"))
            fig.add_trace(go.Scatter(
                x=valid["x"].tolist(), y=valid["y"].tolist(),
                mode="markers",
                marker=dict(size=5, color=ring_col.tolist(), opacity=0.7,
                            line=dict(width=0.5, color="rgba(80,80,80,0.9)")),
                hovertemplate="seg %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                customdata=list(zip(valid["segment"].fillna(0).astype(int),
                                    valid["ring"].fillna("?"))),
                showlegend=False,
            ))

    fig.update_layout(
        **TRANSPARENT_BG,
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, scaleanchor="y",
                   range=[-board_r - 20, board_r + 20]),
        yaxis=dict(visible=False, range=[-board_r - 20, board_r + 20]),
    )
    return fig

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def avg_over_time_chart(stats_df: pd.DataFrame, color: str) -> go.Figure:
    fig = go.Figure()
    if stats_df.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=list(range(1, len(stats_df) + 1)),
        y=stats_df["three_dart_avg"].tolist(),
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
        showlegend=False,
    ))
    fig.update_layout(
        **TRANSPARENT_BG, height=160,
        margin=dict(l=30, r=10, t=10, b=30),
        xaxis=dict(title="Session", gridcolor=GRID_COLOR, color="rgba(80,80,80,0.9)"),
        yaxis=dict(title="3-dart avg", gridcolor=GRID_COLOR, color="rgba(80,80,80,0.9)"),
    )
    return fig


def zone_bar_chart(throws_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if throws_df.empty:
        return fig
    zones = ["triple", "double", "single", "bull", "bullseye", "miss"]
    counts = [int((throws_df["ring"] == z).sum()) for z in zones]
    fig.add_trace(go.Bar(
        y=zones, x=counts, orientation="h",
        marker_color=[ZONE_COLORS[z] for z in zones],
        showlegend=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **TRANSPARENT_BG, height=220,
        margin=dict(l=70, r=10, t=10, b=30),
        xaxis=dict(title="Hits", gridcolor=GRID_COLOR, color="rgba(80,80,80,0.9)"),
        yaxis=dict(color="white"),
    )
    return fig


def cricket_marks_bar(throws_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if throws_df.empty:
        return fig
    cricket_nums = [15, 16, 17, 18, 19, 20, 25]
    labels = ["15", "16", "17", "18", "19", "20", "Bull"]
    totals, hit_rates = [], []
    for num in cricket_nums:
        target_throws = throws_df[throws_df["target_segment"] == num]
        total_at = len(target_throws)
        hits = int(((target_throws["ring"].isin(["single", "double", "triple", "bull", "bullseye"]))).sum())
        totals.append(hits)
        hit_rates.append(hits / total_at * 100 if total_at > 0 else 0)

    colors = []
    for hr in hit_rates:
        if hr > 60:
            colors.append(RING_GREEN)
        elif hr > 35:
            colors.append("#eda100")
        else:
            colors.append(RING_RED)

    fig.add_trace(go.Bar(
        x=labels, y=totals, marker_color=colors,
        showlegend=False,
        hovertemplate="%{x}: %{y} marks (%{customdata:.0f}%)<extra></extra>",
        customdata=hit_rates,
    ))
    fig.update_layout(
        **TRANSPARENT_BG, height=280,
        margin=dict(l=30, r=10, t=10, b=30),
        xaxis=dict(color="rgba(80,80,80,0.9)"),
        yaxis=dict(title="Marks", gridcolor=GRID_COLOR, color="rgba(80,80,80,0.9)"),
    )
    return fig


def cricket_radar(throws_df: pd.DataFrame, color: str) -> go.Figure:
    fig = go.Figure()
    if throws_df.empty:
        return fig
    cricket_nums = [15, 16, 17, 18, 19, 20, 25]
    labels = ["15", "16", "17", "18", "19", "20", "Bull"]
    hit_rates = []
    for num in cricket_nums:
        target_throws = throws_df[throws_df["target_segment"] == num]
        total_at = len(target_throws)
        hits = int((target_throws["ring"].isin(
            ["single", "double", "triple", "bull", "bullseye"])).sum())
        hit_rates.append(hits / total_at * 100 if total_at > 0 else 0)

    # Evenly space 7 labels around 360° to avoid treating "15","16"… as degree values
    n = len(cricket_nums)
    thetas = [i * 360 / n for i in range(n)]

    h = color.lstrip("#")
    fill_rgba = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.4)"
    fig.add_trace(go.Scatterpolar(
        r=hit_rates + [hit_rates[0]],
        theta=thetas + [thetas[0]],
        fill="toself",
        fillcolor=fill_rgba,
        line=dict(color=color, width=2),
        showlegend=False,
    ))
    fig.update_layout(
        **TRANSPARENT_BG, height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], color="rgba(80,80,80,0.8)",
                            gridcolor="rgba(128,128,128,0.3)"),
            angularaxis=dict(
                tickvals=thetas,
                ticktext=labels,
                direction="clockwise",
                rotation=90,
                color="rgba(80,80,80,0.9)",
                gridcolor="rgba(128,128,128,0.3)",
            ),
        ),
    )
    return fig


def comparison_chart(all_stats: pd.DataFrame, selected_player_id: int,
                     player_map: dict) -> go.Figure:
    fig = go.Figure()
    if all_stats.empty:
        return fig
    for i, (pid, grp) in enumerate(all_stats.groupby("player_id")):
        is_selected = (pid == selected_player_id)
        name = player_map.get(pid, str(pid))
        color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(grp) + 1)),
            y=grp["three_dart_avg"].tolist(),
            name=name,
            mode="lines+markers" if is_selected else "lines",
            line=dict(color=color, width=2.5 if is_selected else 1.2,
                      dash="solid" if is_selected else "dot"),
            marker=dict(size=5 if is_selected else 3),
        ))
    fig.update_layout(
        **TRANSPARENT_BG, height=200,
        margin=dict(l=30, r=10, t=10, b=30),
        xaxis=dict(title="Session (last 12)", gridcolor=GRID_COLOR, color="rgba(80,80,80,0.9)"),
        yaxis=dict(title="3-dart avg", gridcolor=GRID_COLOR, color="rgba(80,80,80,0.9)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="rgba(80,80,80,0.9)")),
    )
    return fig

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    init_db()

    players_df = load_players()
    if players_df.empty:
        st.warning("No players found. Run `python scripts/seed_data.py` first.")
        return

    # --- Sidebar ---
    with st.sidebar:
        player_name = st.selectbox("Player", players_df["name"].tolist())
        game_filter_label = st.selectbox("Game type", ["All", "cricket", "501", "301"])
        st.caption("v0.1 — dev mode")

    game_filter = None if game_filter_label == "All" else game_filter_label
    player_id = int(players_df.loc[players_df["name"] == player_name, "id"].iloc[0])
    player_color = PLAYER_COLORS[
        players_df.index[players_df["name"] == player_name].tolist()[0] % len(PLAYER_COLORS)
    ]

    throws_df = load_throws(player_id, game_filter)
    stats_df  = load_session_stats(player_id, game_filter)

    throw_count   = len(throws_df)
    session_count = len(stats_df)

    # --- Header ---
    st.markdown(f"## 🎯 {player_name}")
    st.markdown(f"**{throw_count:,}** throws across **{session_count}** sessions")

    # --- Stat cards ---
    avg_all   = throws_df["score_value"].mean() * 3 if not throws_df.empty else 0
    if not stats_df.empty and len(stats_df) >= 5:
        prev5_avg = stats_df.iloc[-5:]["three_dart_avg"].mean()
        if len(stats_df) >= 10:
            prev_prev5 = stats_df.iloc[-10:-5]["three_dart_avg"].mean()
            delta = round(prev5_avg - prev_prev5, 2)
        else:
            delta = None
    else:
        prev5_avg = avg_all
        delta = None

    triple_pct = (throws_df["ring"] == "triple").mean() * 100 if not throws_df.empty else 0
    miss_pct   = (throws_df["ring"] == "miss").mean()   * 100 if not throws_df.empty else 0
    bull_pct   = (throws_df["ring"].isin(["bull", "bullseye"])).mean() * 100 if not throws_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("3-Dart Avg", f"{prev5_avg:.1f}", delta=f"{delta:+.1f}" if delta is not None else None)
    c2.metric("Triple %",   f"{triple_pct:.1f}%")
    c3.metric("Miss %",     f"{miss_pct:.1f}%")
    c4.metric("Bull %",     f"{bull_pct:.1f}%")

    # --- Two-column layout ---
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Dartboard")
        board_mode = st.radio("Mode", ["scatter", "heat"], horizontal=True, label_visibility="collapsed")
        fig_board = build_dartboard_fig(throws_df, mode=board_mode)
        st.plotly_chart(fig_board, width="stretch", config=NO_MODEBAR)

    with right:
        st.markdown("#### 3-Dart Average Over Time")
        fig_avg = avg_over_time_chart(stats_df, player_color)
        st.plotly_chart(fig_avg, width="stretch", config=NO_MODEBAR)

        st.markdown("#### Hit Zone Breakdown")
        fig_zones = zone_bar_chart(throws_df)
        st.plotly_chart(fig_zones, width="stretch", config=NO_MODEBAR)

    # --- Cricket mark efficiency ---
    st.markdown("#### Cricket Mark Efficiency")
    ca, cb = st.columns(2)
    with ca:
        fig_cricket = cricket_marks_bar(throws_df)
        st.plotly_chart(fig_cricket, width="stretch", config=NO_MODEBAR)
    with cb:
        fig_radar = cricket_radar(throws_df, player_color)
        st.plotly_chart(fig_radar, width="stretch", config=NO_MODEBAR)

    # --- Player comparison ---
    st.markdown("#### Player Comparison (last 12 sessions)")
    all_stats  = load_all_players_session_stats()
    player_map = dict(zip(players_df["id"], players_df["name"]))
    fig_cmp    = comparison_chart(all_stats, player_id, player_map)
    st.plotly_chart(fig_cmp, width="stretch", config=NO_MODEBAR)

    # --- Coaching insights ---
    st.markdown("#### Coaching Insights")
    if not throws_df.empty:
        cricket_nums = [15, 16, 17, 18, 19, 20, 25]
        cnum_labels  = {15: "15", 16: "16", 17: "17", 18: "18",
                        19: "19", 20: "20", 25: "Bull"}
        hit_rates = {}
        for num in cricket_nums:
            t = throws_df[throws_df["target_segment"] == num]
            if len(t) > 0:
                hits = (t["ring"].isin(["single", "double", "triple", "bull", "bullseye"])).sum()
                hit_rates[num] = hits / len(t) * 100

        if hit_rates:
            weakest = min(hit_rates, key=hit_rates.get)
            st.error(f"Weakest Cricket number: **{cnum_labels[weakest]}** "
                     f"({hit_rates[weakest]:.0f}% hit rate) — prioritise practice here.")

        if miss_pct > 15:
            st.warning(f"High miss rate: **{miss_pct:.1f}%** — focus on consistency and reducing spread.")

        if triple_pct < 10:
            st.warning(f"Low triple rate: **{triple_pct:.1f}%** — work on precision targeting for triple rings.")

        if not stats_df.empty and len(stats_df) >= 6:
            recent   = stats_df.iloc[-3:]["three_dart_avg"].mean()
            previous = stats_df.iloc[-6:-3]["three_dart_avg"].mean()
            if recent > previous * 1.03:
                st.success(f"Improving trend — recent avg **{recent:.1f}** vs prior **{previous:.1f}**. Keep it up!")
            elif recent < previous * 0.97:
                st.warning(f"Plateauing or declining — recent avg **{recent:.1f}** vs prior **{previous:.1f}**. Mix up practice routines.")
            else:
                st.info(f"Stable performance — avg **{recent:.1f}**. Try targeting weaker numbers to break through the plateau.")
    else:
        st.info("No throw data available for the selected filter.")


if __name__ == "__main__":
    main()
