"""HIV/AIDS DRC 六腔體模型 —— Streamlit 互動式儀表板。

本檔案只負責「呈現」：所有動力學、R₀、資料生成與參數反演都直接呼叫
:mod:`hiv_drc` 套件，不在這裡重新實作任何計算邏輯。

啟動方式（Windows）::

    .\\.venv\\Scripts\\Activate.ps1
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Streamlit 是無視窗環境，必須在匯入 pyplot 之前切成 Agg 後端。
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter  # noqa: E402

# 若使用者是直接從原始碼目錄執行（尚未 pip install -e .），把 src/ 補進匯入路徑。
# 在 git worktree 中這一步也確保載入的是「本目錄」的 hiv_drc，而非主 checkout 的版本。
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hiv_drc import (  # noqa: E402
    DRC_2020,
    apply,
    estimate_parameters,
    generate_observations,
    initial_state_from_data,
    load_worldbank,
    reproduction_number,
    simulate,
)

# 時變參數之下 R₀ 沒有定義，reproduction_number() 會直接拋錯——這是套件刻意的
# 設計，不是 bug。要看某個時刻凍結下來的瞬時值得走這一個，它不在頂層 __all__ 裡。
from hiv_drc.reproduction import reproduction_number_at  # noqa: E402

# ---------------------------------------------------------------------------
# 常數：標籤、配色與版面設定（純呈現層，與模型無關）
# ---------------------------------------------------------------------------

#: 腔體的中文標籤。人口單位為「百萬人」，時間單位為「年」。
LABELS = {
    "S": "S — 易感者",
    "I1": "I₁ — 感染但未知情",
    "I2": "I₂ — 感染且知情",
    "A": "A — 發病（AIDS）",
    "T": "T — 接受治療",
    "R": "R — 行為改變",
}

#: 固定的「實體 → 顏色」對應。同一個腔體在任何一張圖裡都是同一個顏色，
#: 圖例增減也不會讓顏色重新洗牌。
COLORS = {
    "plhiv": "#4a3aa7",         # 紫 —— 真實數據頁的感染總數
    "art_coverage": "#008300",  # 綠 —— 真實數據頁的治療涵蓋率
    "A": "#2a78d6",   # 藍
    "T": "#eb6834",   # 橘
    "I1": "#1baf7a",  # 青綠
    "I2": "#eda100",  # 黃
    "S": "#4a3aa7",   # 紫
    "R": "#e34948",   # 紅
}

#: 主圖聚焦的四個感染相關腔體。
FOCUS = ("I1", "I2", "A", "T")

#: 潛伏（不可觀測）腔體，預設隱藏。
LATENT = ("S", "R")

GRID_STYLE = {"color": "#d9d8d4", "linewidth": 0.8, "alpha": 0.9}
TEXT_MUTED = "#52514e"

#: 圖上要顯示中文，matplotlib 預設的 DejaVu Sans 沒有 CJK 字符，會變成一排豆腐方塊。
#: 依序找系統裡第一個可用的中文字型（Windows 通常是微軟正黑體）。
CJK_FONTS = (
    "Microsoft JhengHei",  # Windows 繁體
    "Microsoft YaHei",     # Windows 簡體
    "PingFang TC",         # macOS
    "Heiti TC",
    "Noto Sans CJK TC",    # Linux
    "Noto Sans TC",
    "SimHei",
)


def configure_fonts() -> bool:
    """把可用的中文字型排到 matplotlib 的字型序列最前面。

    回傳是否真的找到中文字型。找不到的話（多數 Linux 伺服器、Streamlit Cloud
    的容器都是這樣）圖上的文字必須換成英文——DejaVu Sans 畫不出漢字，
    matplotlib 不會報錯，只會安靜地把每個字換成一個空白方塊。
    """
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in CJK_FONTS if name in available]
    plt.rcParams["font.sans-serif"] = [*chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 減號，避免負號變方塊
    return bool(chosen)


#: 這台機器畫得出中文嗎？網頁文字不受影響（瀏覽器有自己的字型），只有圖會換。
HAS_CJK_FONT = configure_fonts()

#: 圖上用的標籤。兩件事讓它和 LABELS 不同：
#: 中文字型沒有下標字符 ₁ ₂（會變豆腐方塊，而 mathtext 的 $I_1$ 又會把中文
#: 送進數學字型裡消失），所以圖上一律用 ASCII 的 I1/I2；
#: 而沒有中文字型時，整組標籤改用英文。
if HAS_CJK_FONT:
    PLOT_LABELS = {
        **LABELS,
        "I1": "I1 — 感染但未知情",
        "I2": "I2 — 感染且知情",
    }
    PLOT_TEXT = {
        "time": "時間（年）",
        "population": "人口（百萬人）",
        "dynamics_title": "感染相關腔體的時間演變",
        "inverse_title": "合成觀測資料與最小平方擬合",
        "observed": "{label}（觀測）",
        "fitted": "{label}（擬合）",
        # Tab 3：真實數據
        "calendar_year": "年份",
        "plhiv_axis": "感染人數（百萬人）",
        "coverage_axis": "ART 涵蓋率（%）",
        "plhiv_title": "感染總數 —— UNAIDS 實測 vs. 模型",
        "coverage_title": "ART 治療涵蓋率 —— UNAIDS 實測 vs. 模型",
        "real_observed": "UNAIDS 實測",
        "model": "模型模擬",
    }
else:
    PLOT_LABELS = {
        "S": "S — susceptible",
        "I1": "I1 — infected, unaware",
        "I2": "I2 — infected, aware",
        "A": "A — symptomatic (AIDS)",
        "T": "T — on treatment",
        "R": "R — changed behaviour",
    }
    PLOT_TEXT = {
        "time": "Time (years)",
        "population": "Population (millions)",
        "dynamics_title": "Infected compartments over time",
        "inverse_title": "Synthetic observations and the least-squares fit",
        "observed": "{label} (observed)",
        "fitted": "{label} (fitted)",
        # Tab 3
        "calendar_year": "Year",
        "plhiv_axis": "People living with HIV (millions)",
        "coverage_axis": "ART coverage (%)",
        "plhiv_title": "PLHIV — UNAIDS estimates vs. the model",
        "coverage_title": "ART coverage — UNAIDS estimates vs. the model",
        "real_observed": "UNAIDS estimate",
        "model": "Model",
    }

#: 論文基準情境的 R₀，作為滑桿調整後的比較基準。
BASELINE_R0 = float(reproduction_number(DRC_2020).R0)


# ---------------------------------------------------------------------------
# 模型呼叫（全部走 hiv_drc，並加上快取避免滑桿一動就重算）
# ---------------------------------------------------------------------------


def make_parameters(beta: float, alpha: float, phi: float):
    """以側邊欄的三個干預參數覆寫 DRC 2020 基準參數組。"""
    return DRC_2020.replace(beta=beta, alpha=alpha, phi=phi)


@st.cache_data(show_spinner=False)
def cached_r0(beta: float, alpha: float, phi: float) -> dict[str, float]:
    """R₀ 及其三個分項貢獻（R₁/R₂/R₃）。以純量傳參，讓快取鍵單純好雜湊。"""
    r = reproduction_number(make_parameters(beta, alpha, phi))
    return {"R0": float(r.R0), "R1": float(r.R1), "R2": float(r.R2), "R3": float(r.R3),
            "theta": float(r.theta)}


@st.cache_data(show_spinner=False)
def cached_trajectory(
    beta: float, alpha: float, phi: float, years: float
) -> dict[str, np.ndarray]:
    """前向積分 ``years`` 年，回傳 ``t`` 與六個腔體的時間序列（百萬人）。"""
    solution = simulate(make_parameters(beta, alpha, phi), t_span=(0.0, years))
    series = {name: solution[name] for name in LABELS}
    series["t"] = solution.t
    series["infected"] = solution.infected
    series["prevalence"] = solution.prevalence
    return series


@st.cache_data(show_spinner=False)
def cached_observations(beta: float, alpha: float, phi: float, noise: float, seed: int):
    """由目前參數生成帶高斯雜訊的合成觀測資料（A 與 T）。"""
    return generate_observations(
        p=make_parameters(beta, alpha, phi), noise=noise, seed=seed
    )


# -- Tab 3：真實數據與時變參數 ----------------------------------------------

#: UNAIDS／World Bank 的 DRC 快照。load_worldbank 的預設路徑是相對的，
#: 而 Streamlit 的工作目錄未必是專案根目錄，所以這裡自己組絕對路徑。
REAL_DATA_PATH = Path(__file__).resolve().parent / "data" / "real" / "drc_worldbank.csv"


@st.cache_data(show_spinner=False)
def cached_real_data(first_year: int):
    """讀入真實觀測（plhiv 與 art_coverage）與年份、總人口等環境變數。

    這份資料是committed 的 CSV 快照，不連網路——下載是 scripts/fetch_worldbank.py
    的事，這樣同一份圖表任何時候重跑都會得到同一個答案。
    """
    return load_worldbank(path=REAL_DATA_PATH, first_year=first_year)


def scaleup_parameters(
    beta: float,
    alpha: float,
    phi: float,
    enabled: bool,
    alpha_ceiling: float,
    alpha_midpoint: float,
    alpha_rate: float,
    lam_ceiling: float,
    lam_midpoint: float,
    lam_rate: float,
):
    """側邊欄的常數參數，加上（可選的）α 與 λ logistic 擴展。

    ``enabled`` 為 False 時完全不碰那些欄位，參數組維持常係數——也就是論文
    原始的模型，連 R₀ 與平衡點的理論都仍然適用。
    """
    p = make_parameters(beta, alpha, phi)
    if not enabled:
        return p
    return p.replace(
        alpha_ceiling=alpha_ceiling,
        alpha_midpoint=alpha_midpoint,
        alpha_rate=alpha_rate,
        lam_ceiling=lam_ceiling,
        lam_midpoint=lam_midpoint,
        lam_rate=lam_rate,
    )


@st.cache_data(show_spinner=False)
def cached_real_simulation(
    beta: float,
    alpha: float,
    phi: float,
    enabled: bool,
    alpha_ceiling: float,
    alpha_midpoint: float,
    alpha_rate: float,
    lam_ceiling: float,
    lam_midpoint: float,
    lam_rate: float,
    first_year: int,
) -> dict[str, np.ndarray]:
    """從資料的第一年出發積分，回傳與觀測同名的兩條模型曲線。

    初始狀態不用論文 Table 2，而是用 initial_state_from_data 從當年的
    PLHIV、ART 涵蓋率與總人口重建——Table 2 標著 2020，但它的治療腔體
    對應的是 2013 年的涵蓋率，而 T 正是識別 α 的那條序列。
    """
    observations, context = cached_real_data(first_year)
    y0 = initial_state_from_data(
        plhiv=float(observations.values["plhiv"][0]),
        art_coverage=float(observations.values["art_coverage"][0]),
        population=float(context["population"][0]),
    )
    p = scaleup_parameters(
        beta, alpha, phi, enabled,
        alpha_ceiling, alpha_midpoint, alpha_rate,
        lam_ceiling, lam_midpoint, lam_rate,
    )
    span = (0.0, float(observations.t[-1]))
    solution = simulate(p, y0=y0, t_span=span, n_points=401)
    # apply() 走的是套件的觀測算子登錄表，所以 plhiv 與 art_coverage 的定義
    # 和資料端、擬合端完全是同一份，不會在這裡分岔。
    modelled = apply(solution, observations.names)
    return {"t": solution.t, **{name: modelled[name] for name in observations.names},
            "y0": y0}


# ---------------------------------------------------------------------------
# 繪圖工具
# ---------------------------------------------------------------------------


def style_axes(ax, xlabel: str, ylabel: str) -> None:
    """統一的座標軸樣式：收斂的格線、無上右框線、次要色的軸標籤。"""
    ax.grid(True, **GRID_STYLE)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c4")
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=10)


def plot_trajectory(series: dict[str, np.ndarray], show_latent: bool, log_scale: bool):
    """疫情軌跡圖：主圖只畫 I₁/I₂/A/T；S 與 R 另外畫在下方獨立面板。

    刻意不用雙 y 軸——S 約 88 百萬、A 約 0.05 百萬，兩條尺度硬塞進同一張圖
    只會讓斜率無法互相比較。要看 S 與 R 時，改成共用時間軸的第二個面板。
    """
    if show_latent:
        fig, (ax, ax_latent) = plt.subplots(
            2, 1, figsize=(9, 6.2), sharex=True, height_ratios=[2, 1]
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax_latent = None

    t = series["t"]
    for name in FOCUS:
        ax.plot(t, series[name], label=PLOT_LABELS[name], color=COLORS[name], linewidth=2)
    if log_scale:
        ax.set_yscale("log")
        # 對數刻度的預設標籤是 mathtext（$10^{-1}$），其中的 U+2212 減號在中文
        # 字型裡不存在，會印出一個空心方塊。改用純文字格式，避開整條 mathtext 路徑。
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    style_axes(
        ax,
        "" if ax_latent is not None else PLOT_TEXT["time"],
        PLOT_TEXT["population"],
    )
    ax.set_title(PLOT_TEXT["dynamics_title"], color="#0b0b0b", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_MUTED, ncols=2)

    if ax_latent is not None:
        for name in LATENT:
            ax_latent.plot(
                t, series[name], label=PLOT_LABELS[name], color=COLORS[name], linewidth=2
            )
        style_axes(ax_latent, PLOT_TEXT["time"], PLOT_TEXT["population"])
        ax_latent.legend(frameon=False, fontsize=9, labelcolor=TEXT_MUTED, ncols=2)

    fig.tight_layout()
    return fig


def plot_inverse(observations, fit_result):
    """反問題圖：觀測散點 +（若已擬合）最佳擬合曲線。"""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name in observations.names:
        ax.scatter(
            observations.t,
            observations.values[name],
            s=30,
            color=COLORS[name],
            edgecolor="#ffffff",
            linewidth=0.8,
            zorder=3,
            label=PLOT_TEXT["observed"].format(label=PLOT_LABELS[name]),
        )
    if fit_result is not None:
        solution = fit_result.solution
        for name in observations.names:
            ax.plot(
                solution.t,
                solution[name],
                color=COLORS[name],
                linewidth=2,
                zorder=2,
                label=PLOT_TEXT["fitted"].format(label=PLOT_LABELS[name]),
            )
    style_axes(ax, PLOT_TEXT["time"], PLOT_TEXT["population"])
    ax.set_title(PLOT_TEXT["inverse_title"], color="#0b0b0b", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_MUTED, ncols=2)
    fig.tight_layout()
    return fig


def plot_real_data(years, observed, model_years, model, key: str, percent: bool):
    """真實觀測（散點）與模型曲線（折線）疊在同一組座標軸上。

    一張圖只有一個量，觀測與模型用「點 vs. 線」區分而不是用兩種顏色——
    它們是同一件事的兩個來源，不是兩個不同的序列。
    """
    scale = 100.0 if percent else 1.0
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(
        model_years,
        np.asarray(model) * scale,
        color=COLORS[key],
        linewidth=2,
        zorder=2,
        label=PLOT_TEXT["model"],
    )
    ax.scatter(
        years,
        np.asarray(observed) * scale,
        s=32,
        color=TEXT_MUTED,
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=3,
        label=PLOT_TEXT["real_observed"],
    )
    style_axes(
        ax,
        PLOT_TEXT["calendar_year"],
        PLOT_TEXT["coverage_axis"] if percent else PLOT_TEXT["plhiv_axis"],
    )
    ax.set_title(
        PLOT_TEXT["coverage_title"] if percent else PLOT_TEXT["plhiv_title"],
        color="#0b0b0b",
        fontsize=12,
        loc="left",
    )
    # 橫軸是年份，刻度必須落在整數年上：預設的 locator 會挑 2.5 年的間隔，
    # 再被格式化成整數，於是 2007.5 顯示成「2008」——看起來合理，其實是錯的。
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    if percent:
        ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_MUTED)
    fig.tight_layout()
    return fig


def fmt(value: float, digits: int = 6) -> str:
    """把可能是 nan/inf 的數字排版成表格看得懂的樣子。"""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def ci_table_markdown(fit_result) -> str:
    """把 FitResult 的 Wald 95% 信賴區間排成 Markdown 表格。"""
    has_truth = fit_result.truth is not None
    header = "| 參數 | 估計值 | 標準誤 | Wald 95% 信賴區間 |"
    divider = "| --- | ---: | ---: | :---: |"
    if has_truth:
        header += " 真值 | 相對誤差 | 涵蓋真值 |"
        divider += " ---: | ---: | :---: |"
    rows = [header, divider]

    relative = fit_result.relative_errors() if has_truth else {}
    covers = fit_result.covers_truth() if has_truth else {}
    for name in fit_result.names:
        low, high = fit_result.ci95[name]
        row = (
            f"| `{name}` | {fmt(fit_result.estimates[name])} "
            f"| {fmt(fit_result.stderr[name])} "
            f"| [{fmt(low)}, {fmt(high)}] |"
        )
        if has_truth:
            covered = "✅" if covers[name] else "❌"
            row += (
                f" {fmt(fit_result.truth[name])} "
                f"| {relative[name]:+.2f}% | {covered} |"
            )
        rows.append(row)
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 版面：側邊欄
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HIV/AIDS DRC 動力學儀表板", page_icon="🧬", layout="wide"
)

#: 側邊欄滑桿的預設值（DRC 2020 基準情境）。
DEFAULTS = {"beta": 0.15, "alpha": 0.035, "phi": 0.083, "years": 50}


# 預設值只在第一次進站時寫進 session_state，之後 widget 全部以 key 讀寫。
# 若同時給 value= 又用 session_state 設值，Streamlit 每次 rerun 都會警告兩者衝突。
for _key, _default in DEFAULTS.items():
    st.session_state.setdefault(_key, _default)


def reset_to_baseline() -> None:
    """把三個干預參數與模擬年限復原成基準值。

    以 callback 形式執行：Streamlit 會在下一次 rerun 之前套用，
    才不會撞上「widget 已實例化後不得改寫其 session_state」的限制。
    """
    st.session_state.update(DEFAULTS)


with st.sidebar:
    st.title("🧬 HIV/AIDS 傳播動力學")
    st.caption(
        "剛果民主共和國（DRC）六腔體 ODE 模型 —— "
        "Mbayi, Mpompi & Munyakazi, *Tamkang Journal of Mathematics* 57(2), "
        "149–169 (2026)。"
    )
    st.markdown(
        "狀態向量為 **S → I₁ → I₂ → A → T → R**，人口以百萬人計、時間以年計。"
        "下方三個滑桿分別代表傳播、治療與預防三種介入手段；"
        "所有計算皆由已完成單元測試的 `hiv_drc` 套件負責。"
    )

    st.subheader("干預參數")
    beta = st.slider(
        "β — 有效接觸率 (Effective contact rate)",
        min_value=0.0, max_value=0.3, step=0.001, format="%.3f", key="beta",
        help="每年每人的有效傳播接觸強度。降低 β 對應保險套推廣、衛教等一級預防。",
    )
    alpha = st.slider(
        "α — 治療涵蓋率 (Treatment uptake)",
        min_value=0.0, max_value=0.1, step=0.001, format="%.3f", key="alpha",
        help="知情感染者 I₂ 每年進入治療 T 的速率。提高 α 對應擴大 ART 服務量能。",
    )
    phi = st.slider(
        "φ — 保護行為採納率 (Protective behaviour)",
        min_value=0.0, max_value=0.2, step=0.001, format="%.3f", key="phi",
        help="易感者採取保護行為而離開 S 進入 R 的速率。",
    )

    st.divider()
    years = st.slider(
        "模擬年限（年）", min_value=10, max_value=100, step=5, key="years",
    )
    st.button("↺ 回到 DRC 2020 基準值", on_click=reset_to_baseline)

parameters = make_parameters(beta, alpha, phi)

# ---------------------------------------------------------------------------
# 版面：主畫面
# ---------------------------------------------------------------------------

tab_dynamics, tab_inverse, tab_real = st.tabs(
    ["📈 疫情軌跡與公衛沙盒", "🔍 反問題與參數估計", "🌍 真實數據與政策推廣"]
)

# --- Tab 1：疫情軌跡 --------------------------------------------------------

with tab_dynamics:
    r0 = cached_r0(beta, alpha, phi)
    series = cached_trajectory(beta, alpha, phi, float(years))

    col_r0, col_infected, col_art = st.columns(3)
    with col_r0:
        # 停在基準值時不顯示 delta，免得畫面上掛一個沒有意義的 +0.0000。
        gap = r0["R0"] - BASELINE_R0
        st.metric(
            "基本再生數 R₀",
            f"{r0['R0']:.4f}",
            delta=f"{gap:+.4f} vs. 基準" if abs(gap) > 5e-5 else None,
            delta_color="inverse",  # R₀ 上升是壞消息，用反向配色
            help="由 hiv_drc.reproduction_number 以論文式 (3.9)–(3.12) 的封閉解計算。",
        )
    infected_end = float(series["infected"][-1])
    infected_start = float(series["infected"][0])
    with col_infected:
        st.metric(
            f"第 {years} 年感染人數（百萬）",
            f"{infected_end:.4f}",
            delta=f"{infected_end - infected_start:+.4f}",
            delta_color="inverse",
        )
    with col_art:
        on_treatment = float(series["T"][-1])
        coverage = on_treatment / infected_end if infected_end > 0 else float("nan")
        st.metric(
            f"第 {years} 年治療涵蓋率",
            f"{coverage:.1%}" if np.isfinite(coverage) else "—",
            help="T /（I₁ + I₂ + A + T）。",
        )

    if r0["R0"] < 1.0:
        st.success(
            f"✅ **R₀ = {r0['R0']:.4f} < 1** —— 無病平衡點在此參數組下為局部漸近穩定，"
            "疫情長期趨於消滅。"
        )
    else:
        st.warning(
            f"⚠️ **R₀ = {r0['R0']:.4f} ≥ 1** —— 疫情可自我維持，"
            "模型會收斂到地方性流行（endemic）平衡點。"
        )

    with st.expander("R₀ 的分項貢獻（各感染階段的次世代貢獻）"):
        theta = r0["theta"]
        st.markdown(
            f"""
R₀ = β · θ · (R₁ + R₂ + R₃)，其中 θ = μ/(μ+φ) = **{theta:.4f}** 為無病平衡下的易感比例。

| 分項 | 來源腔體 | 數值 | 佔 R₀ 比重 |
| --- | --- | ---: | ---: |
| R₁ | I₁（未知情） | {r0['R1']:.4f} | {r0['R1'] / (r0['R1'] + r0['R2'] + r0['R3']):.1%} |
| R₂ | I₂（知情） | {r0['R2']:.4f} | {r0['R2'] / (r0['R1'] + r0['R2'] + r0['R3']):.1%} |
| R₃ | A（發病） | {r0['R3']:.4f} | {r0['R3'] / (r0['R1'] + r0['R2'] + r0['R3']):.1%} |

φ 只透過 θ 影響 R₀，α 則同時改變 I₂ 的停留時間與後續路徑，因此兩者的邊際效果並不對稱。
            """
        )

    st.subheader("疫情軌跡")
    opt_latent, opt_log = st.columns(2)
    show_latent = opt_latent.checkbox(
        "同時顯示 S 與 R（獨立面板）",
        value=False,
        help="S 約 88 百萬、A 約 0.05 百萬，量級差三個數量級；分成兩個共用時間軸的面板"
        "比擠在同一組 y 軸上更能看出各自的斜率。",
    )
    log_scale = opt_log.checkbox("感染腔體使用對數座標", value=False)

    figure = plot_trajectory(series, show_latent, log_scale)
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("以表格檢視（每 5 年取樣）"):
        step = max(1, len(series["t"]) // max(1, int(years / 5)))
        table = {"時間（年）": np.round(series["t"][::step], 2)}
        table.update(
            {LABELS[name]: np.round(series[name][::step], 5) for name in FOCUS + LATENT}
        )
        st.dataframe(table, hide_index=True)

# --- Tab 2：反問題 ----------------------------------------------------------

with tab_inverse:
    st.markdown(
        "只有 **A**（發病人數）與 **T**（治療人數）是監測系統實際會回報的量；"
        "S、I₁、I₂、R 皆為潛伏變數。這裡由目前的參數組生成帶 5% 高斯雜訊的合成觀測，"
        "再交給 `estimate_parameters` 以有界非線性最小平方法回推參數，"
        "看看它能不能把「它從未被告知」的真值找回來。"
    )

    control_left, control_mid, control_right = st.columns([1, 1, 2])
    noise = control_left.slider(
        "量測雜訊 η", min_value=0.0, max_value=0.20, value=0.05, step=0.01,
        format="%.2f", help="相對雜訊水準；0.05 即 5%。",
    )
    seed = control_mid.number_input(
        "隨機種子", min_value=0, max_value=99_999_999, value=20260830, step=1,
        help="固定種子讓每次生成的觀測資料可以重現。",
    )
    fit_names = control_right.multiselect(
        "要估計的參數",
        options=["beta", "alpha", "phi"],
        default=["beta", "alpha"],
        help="未被選中的參數會固定在側邊欄設定的值上（即 baseline）。",
    )

    button_left, button_right = st.columns(2)
    generate_clicked = button_left.button(
        "🎲 生成模擬觀測數據", type="secondary"
    )
    fit_clicked = button_right.button(
        "📐 執行最小平方法擬合", type="primary"
    )

    # 目前的情境指紋：參數一改，舊的觀測與擬合就不再對應，必須失效。
    scenario = (beta, alpha, phi, float(noise), int(seed))
    if st.session_state.get("scenario") != scenario:
        st.session_state["scenario"] = scenario
        st.session_state.pop("observations", None)
        st.session_state.pop("fit", None)

    if generate_clicked:
        with st.spinner("正在以目前參數積分並加入量測誤差…"):
            st.session_state["observations"] = cached_observations(
                beta, alpha, phi, float(noise), int(seed)
            )
        st.session_state.pop("fit", None)  # 觀測換了，舊擬合作廢

    if fit_clicked:
        if "observations" not in st.session_state:
            st.error("請先按「生成模擬觀測數據」，才有資料可以擬合。")
        elif not fit_names:
            st.error("請至少選擇一個要估計的參數。")
        else:
            with st.spinner(
                "正在執行有界非線性最小平方法…（每次目標函數評估都要積分一次六腔體系統，請稍候）"
            ):
                st.session_state["fit"] = estimate_parameters(
                    st.session_state["observations"],
                    fit=tuple(fit_names),
                    baseline=parameters,
                )

    observations = st.session_state.get("observations")
    fit_result = st.session_state.get("fit")

    if observations is None:
        st.info("尚未生成觀測資料。按下上方「🎲 生成模擬觀測數據」開始。")
    else:
        figure = plot_inverse(observations, fit_result)
        st.pyplot(figure)
        plt.close(figure)
        st.caption(
            f"{observations.n_points} 個年度觀測點，"
            f"雜訊模型：{observations.noise_model}（η = {observations.noise:.0%}）。"
        )

    if fit_result is not None:
        st.subheader("參數估計結果")
        if not fit_result.success:
            st.warning(f"最佳化器回報未收斂：{fit_result.message}")
        st.markdown(ci_table_markdown(fit_result))

        rmse = "、".join(
            f"{LABELS[name]}：{value:.6f}" for name, value in fit_result.rmse.items()
        )
        st.caption(
            f"成本函數 = {fit_result.cost:.6e}　|　"
            f"RMSE（百萬人）{rmse}　|　"
            f"殘差評估次數 = {fit_result.nfev}"
        )

        col_r0_fit, col_corr = st.columns(2)
        col_r0_fit.metric(
            "擬合參數所隱含的 R₀",
            f"{fit_result.R0:.4f}",
            delta=f"{fit_result.R0 - cached_r0(beta, alpha, phi)['R0']:+.4f} vs. 真值",
            delta_color="off",
        )
        if len(fit_result.names) > 1:
            correlation = fit_result.correlation[0, 1]
            col_corr.metric(
                f"corr({fit_result.names[0]}, {fit_result.names[1]})",
                fmt(float(correlation), 4),
                help="接近 ±1 代表這兩個參數在資料中無法被分開，"
                "殘差再小也不表示估計值可信。",
            )
        if any(
            not np.isfinite(value) for value in fit_result.stderr.values()
        ):
            st.warning(
                "有參數的標準誤為 inf 或 nan：Jacobian 在該方向上秩虧，"
                "資料完全無法約束它（例如 β 被推到下界 0，感染項整個消失）。"
                "此時的 Wald 區間不具意義。"
            )

# --- Tab 3：真實數據與政策推廣 --------------------------------------------

with tab_real:
    st.markdown(
        "前兩頁的資料都是模型自己生成的，答案早就知道。這一頁換成 **UNAIDS／World Bank "
        "實際發布的剛果民主共和國序列**：感染總數 `plhiv` 與治療涵蓋率 `art_coverage`。"
        "兩者都是套件裡登錄好的觀測算子，不是單一腔體——沒有任何一個腔體對應「涵蓋率」。"
    )

    first_year = st.slider(
        "資料起始年份", min_value=1990, max_value=2015, value=2005, step=1,
        key="first_year",
        help="ART 涵蓋率要到 2000 年才有發布，而且在 2000 年代中期以前近乎為零。",
    )

    try:
        observations, context = cached_real_data(int(first_year))
    except (FileNotFoundError, ValueError) as error:
        st.error(f"讀不到真實數據：{error}")
        st.stop()

    years = np.asarray(context["year"])
    coverage_observed = np.asarray(observations.values["art_coverage"])

    st.warning(
        f"**真實世界的 ART 涵蓋率從 {years[0]:.0f} 年的 "
        f"{coverage_observed[0]:.0%} 一路衝到 {years[-1]:.0f} 年的 "
        f"{coverage_observed[-1]:.0%}。** 常係數模型畫不出這種 S 形曲線——它的 α "
        "是一個固定不動的速率，只能給出單調趨近某個水平的軌跡。這正是引入時間變動"
        "參數（time-varying rates）的理由：把政策推廣本身放進模型裡。"
    )

    st.subheader("時間變動參數沙盒")
    scaleup_on = st.checkbox(
        "啟用 Logistic 政策擴展（取消勾選＝常數參數，論文原始設定）",
        value=False,
        key="scaleup_on",
        help="關閉時 α 與 λ 都是常數，系統維持自治，R₀ 與平衡點的理論仍然成立；"
        "開啟後兩者隨時間走 logistic 曲線，系統變成非自治的。",
    )

    col_alpha, col_lam = st.columns(2)
    with col_alpha:
        st.markdown("**α(t) —— 治療推廣**")
        alpha_ceiling = st.slider(
            "α 上限（ceiling）", min_value=0.0, max_value=2.0, value=0.2, step=0.01,
            key="alpha_ceiling", disabled=not scaleup_on,
            help="推廣結束後 α 穩定下來的速率。預設值刻意調成讓 2024 年的涵蓋率"
            "幾乎正中實測的 71% —— 然後看看感染總數差多少。",
        )
        alpha_midpoint = st.slider(
            "α 推廣中點（年）", min_value=0.0, max_value=40.0, value=10.0, step=0.5,
            key="alpha_midpoint", disabled=not scaleup_on,
            help="從資料起始年算起第幾年通過半程。",
        )
        alpha_rate = st.slider(
            "α 推廣速率", min_value=0.0, max_value=2.0, value=0.4, step=0.05,
            key="alpha_rate", disabled=not scaleup_on,
            help="數字越大，政策轉折越陡。",
        )
    with col_lam:
        st.markdown("**λ(t) —— 篩檢診斷推廣**")
        lam_ceiling = st.slider(
            "λ 上限（ceiling）", min_value=0.0, max_value=2.0, value=0.3, step=0.01,
            key="lam_ceiling", disabled=not scaleup_on,
            help="治療追不上診斷：α 把人帶出 I₂，但只有 λ 能把人放進去。",
        )
        lam_midpoint = st.slider(
            "λ 推廣中點（年）", min_value=0.0, max_value=40.0, value=8.0, step=0.5,
            key="lam_midpoint", disabled=not scaleup_on,
        )
        lam_rate = st.slider(
            "λ 推廣速率", min_value=0.0, max_value=2.0, value=0.4, step=0.05,
            key="lam_rate", disabled=not scaleup_on,
        )

    with st.spinner("正在以真實初始狀態積分…"):
        simulated = cached_real_simulation(
            beta, alpha, phi, bool(scaleup_on),
            float(alpha_ceiling), float(alpha_midpoint), float(alpha_rate),
            float(lam_ceiling), float(lam_midpoint), float(lam_rate),
            int(first_year),
        )
    model_years = years[0] + simulated["t"]

    # 末年的模型值 vs. 實測值——把「差多少」講成數字，而不是只讓人看線。
    model_coverage_end = float(np.interp(observations.t[-1], simulated["t"],
                                         simulated["art_coverage"]))
    model_plhiv_end = float(np.interp(observations.t[-1], simulated["t"],
                                      simulated["plhiv"]))
    metric_cov, metric_plhiv, metric_r0 = st.columns(3)
    metric_cov.metric(
        f"{years[-1]:.0f} 年 ART 涵蓋率（模型）",
        f"{model_coverage_end:.1%}",
        delta=f"{(model_coverage_end - coverage_observed[-1]) * 100:+.1f} 個百分點 vs. 實測",
        delta_color="off",
    )
    metric_plhiv.metric(
        f"{years[-1]:.0f} 年感染總數（模型）",
        f"{model_plhiv_end:.4f}",
        delta=f"{model_plhiv_end - float(observations.values['plhiv'][-1]):+.4f} vs. 實測",
        delta_color="off",
    )
    parameters_real = scaleup_parameters(
        beta, alpha, phi, bool(scaleup_on),
        float(alpha_ceiling), float(alpha_midpoint), float(alpha_rate),
        float(lam_ceiling), float(lam_midpoint), float(lam_rate),
    )
    r0_start = reproduction_number_at(parameters_real, 0.0).R0
    r0_end = reproduction_number_at(parameters_real, float(observations.t[-1])).R0
    metric_r0.metric(
        "R₀(t) 起點 → 終點",
        f"{r0_start:.3f} → {r0_end:.3f}",
        help="時變參數下 R₀ 不是門檻定理，只是「此刻政策推得多用力」的診斷值："
        "系統從來沒有在任何一個凍結狀態停留夠久，讓對應的平衡點發生作用。",
    )

    figure = plot_real_data(
        years, observations.values["plhiv"], model_years, simulated["plhiv"],
        key="plhiv", percent=False,
    )
    st.pyplot(figure)
    plt.close(figure)

    figure = plot_real_data(
        years, coverage_observed, model_years, simulated["art_coverage"],
        key="art_coverage", percent=True,
    )
    st.pyplot(figure)
    plt.close(figure)

    with st.expander("初始狀態是怎麼來的？（以及哪些部分是假設）"):
        y0 = simulated["y0"]
        st.markdown(
            f"""
初始狀態不取論文 Table 2，而是由 `initial_state_from_data` 從 {years[0]:.0f} 年的實測值重建：
感染總數 {float(observations.values['plhiv'][0]):.4f} 百萬、ART 涵蓋率
{coverage_observed[0]:.1%}、總人口 {float(context['population'][0]):.2f} 百萬。

| 腔體 | S | I₁ | I₂ | A | T | R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 百萬人 | {y0[0]:.4f} | {y0[1]:.4f} | {y0[2]:.4f} | {y0[3]:.4f} | {y0[4]:.4f} | {y0[5]:.4f} |

資料只釘得住三個數字（總人口、感染總數、涵蓋率），模型卻需要六個。**未治療感染者
如何拆成 I₁／I₂／A、以及 R 有多少人，是假設而不是量測**——套件把它們做成
`untreated_split` 與 `behaviour_changed` 兩個明擺著的參數，這裡用的是論文的預設拆法。

順帶一提，Table 2 標示為 2020 年，但它的 ART 涵蓋率 15.9% 其實對應 2013 年（14%），
而 T 正是識別治療速率 α 的那條序列——從 Table 2 出發做真實數據擬合，等於把這個誤差
直接灌進你想估計的那個參數裡。
            """
        )

    st.info(
        "把上面的核取方塊打開再關掉，比較兩條曲線：常數 α 只能單調趨近某個水平，"
        "而實測的涵蓋率是先慢、後急、再平的 S 形。\n\n"
        "**但請同時看兩張圖。** 預設的 logistic 參數讓 2024 年的 ART 涵蓋率幾乎"
        "正中實測值，而同一組參數下的感染總數只有實測的三分之一左右——**對上一條序列"
        "不等於模型是對的**。README 的「Meeting real data」與「Time-varying rates」"
        "兩節記錄了完整的量化結果，包括那個負面發現：光讓 α 隨時間變還不夠，"
        "因為瓶頸其實在診斷率 λ 而不是治療率——治不了還沒被診斷出來的人。"
    )

st.divider()
st.caption(
    "所有動力學、R₀、資料生成與參數反演均由 `hiv_drc` 套件提供；"
    "本檔案只負責互動與呈現。"
)
