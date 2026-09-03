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
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# 若使用者是直接從原始碼目錄執行（尚未 pip install -e .），把 src/ 補進匯入路徑。
# 在 git worktree 中這一步也確保載入的是「本目錄」的 hiv_drc，而非主 checkout 的版本。
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hiv_drc import (  # noqa: E402
    DRC_2020,
    estimate_parameters,
    generate_observations,
    reproduction_number,
    simulate,
)

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

#: 圖上用的標籤。中文字型（如微軟正黑體）沒有下標字符 ₁ ₂，畫出來會變成豆腐方塊；
#: mathtext（$I_1$）雖然排得出下標，卻會把整段文字丟給數學字型，中文反而不見。
#: 圖裡因此一律用 ASCII 的 I1/I2，網頁文字則保留漂亮的下標。
PLOT_LABELS = {
    **LABELS,
    "I1": "I1 — 感染但未知情",
    "I2": "I2 — 感染且知情",
}

#: 固定的「實體 → 顏色」對應。同一個腔體在任何一張圖裡都是同一個顏色，
#: 圖例增減也不會讓顏色重新洗牌。
COLORS = {
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


def configure_fonts() -> None:
    """把可用的中文字型排到 matplotlib 的字型序列最前面。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = [name for name in CJK_FONTS if name in available]
    plt.rcParams["font.sans-serif"] = [*chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 減號，避免負號變方塊


configure_fonts()

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
    style_axes(ax, "" if ax_latent is not None else "時間（年）", "人口（百萬人）")
    ax.set_title("感染相關腔體的時間演變", color="#0b0b0b", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_MUTED, ncols=2)

    if ax_latent is not None:
        for name in LATENT:
            ax_latent.plot(
                t, series[name], label=PLOT_LABELS[name], color=COLORS[name], linewidth=2
            )
        style_axes(ax_latent, "時間（年）", "人口（百萬人）")
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
            label=f"{PLOT_LABELS[name]}（觀測）",
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
                label=f"{PLOT_LABELS[name]}（擬合）",
            )
    style_axes(ax, "時間（年）", "人口（百萬人）")
    ax.set_title("合成觀測資料與最小平方擬合", color="#0b0b0b", fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_MUTED, ncols=2)
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

tab_dynamics, tab_inverse = st.tabs(
    ["📈 疫情軌跡與公衛沙盒", "🔍 反問題與參數估計"]
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

st.divider()
st.caption(
    "所有動力學、R₀、資料生成與參數反演均由 `hiv_drc` 套件提供；"
    "本檔案只負責互動與呈現。"
)
