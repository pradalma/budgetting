#!/usr/bin/env python3
import io
import os
import math
from dataclasses import dataclass
from datetime import date
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# Matplotlib (no seaborn)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.backends.backend_pdf import PdfPages

# ---------- Budget logic ----------

FREQ_TO_MONTHLY = {
    "monthly":   1.0,
    "biweekly":  26.0 / 12.0,
    "weekly":    52.0 / 12.0,
    "quarterly": 1.0 / 3.0,
    "annual":    1.0 / 12.0,
    "once":      1.0,  # interpreted as a one-time expense in this month
}
FREQ_CHOICES = list(FREQ_TO_MONTHLY.keys())

@dataclass
class LineItem:
    name: str
    amount: float
    frequency: str
    def monthly_amount(self) -> float:
        return self.amount * FREQ_TO_MONTHLY.get(self.frequency, 1.0)

def money(x: float) -> str:
    return f"${x:,.2f}"

def percent(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole else 0.0

def round2(x: float) -> float:
    return math.floor(x * 100 + 0.5) / 100.0

# ---------- Chart ----------

def make_expense_pie(expenses: List[LineItem],
                     small_pct: float = 0.03,
                     min_sep: float = 0.06) -> Optional[plt.Figure]:
    sizes_all = [e.monthly_amount() for e in expenses if e.monthly_amount() > 0]
    labels_all = [e.name for e in expenses if e.monthly_amount() > 0]
    if not sizes_all:
        return None

    total = float(sum(sizes_all))
    annotated, tiny = [], []
    for lbl, val in zip(labels_all, sizes_all):
        pct = val / total
        (annotated if pct >= small_pct else tiny).append((lbl, val, pct))

    fig, ax = plt.subplots(figsize=(7, 7))
    all_sizes = [v for _, v, _ in annotated] + [v for _, v, _ in tiny]
    wedges, _ = ax.pie(all_sizes, labels=None, startangle=90)

    wedge_iter = iter(wedges)
    ann_wedges  = [next(wedge_iter) for _ in annotated]
    tiny_wedges = [next(wedge_iter) for _ in tiny]

    ann_specs = []
    for (lbl, val, pct), w in zip(annotated, ann_wedges):
        ang = 0.5 * (w.theta2 + w.theta1)
        ang_rad = np.deg2rad(ang)
        x, y = np.cos(ang_rad), np.sin(ang_rad)
        ha = "left" if x >= 0 else "right"
        text = f"{lbl}: {val:.2f} ({pct:.1%})"
        ann_specs.append([x, y, ha, ang, text])

    def spread(specs: list):
        specs.sort(key=lambda t: t[1])
        for i in range(1, len(specs)):
            if specs[i][1] - specs[i-1][1] < min_sep:
                specs[i][1] = specs[i-1][1] + min_sep

    left  = [s for s in ann_specs if s[0] < 0]
    right = [s for s in ann_specs if s[0] >= 0]
    spread(left); spread(right)
    placed = left + right

    kw = dict(arrowprops=dict(arrowstyle="-", lw=0.8), va="center", fontsize=9)
    for x, y, ha, ang, text in placed:
        tx, ty = 1.25 * np.sign(x), 1.25 * y
        kw["arrowprops"].update({"connectionstyle": f"angle,angleA=0,angleB={ang}"})
        ann = ax.annotate(text, xy=(x, y), xytext=(tx, ty), ha=ha, **kw)
        ann.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])

    for (lbl, val, pct), w in zip(tiny, tiny_wedges):
        ang = 0.5 * (w.theta2 + w.theta1)
        ang_rad = np.deg2rad(ang)
        x, y = 0.70 * np.cos(ang_rad), 0.70 * np.sin(ang_rad)
        ax.text(x, y, f"{pct:.1%}", ha="center", va="center", fontsize=8,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

    if tiny:
        from matplotlib.patches import Patch
        legend_handles = []
        for (lbl, val, pct), w in zip(tiny, tiny_wedges):
            color = w.get_facecolor()
            legend_handles.append(Patch(facecolor=color, edgecolor='none',
                                        label=f"{lbl}: {money(val)} ({pct:.1%})"))
        ax.legend(handles=legend_handles, title="Smaller categories",
                  loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=9, ncol=1)

    ax.set_title("Monthly Expenses Breakdown")
    ax.axis("equal")
    plt.tight_layout()
    return fig

# ---------- CSV + PDF ----------

def make_csv_bytes(incomes: List[LineItem], expenses: List[LineItem]) -> bytes:
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["type","name","frequency","entered_amount","monthly_amount"])
    for it in incomes:
        w.writerow(["income", it.name, it.frequency, f"{it.amount:.2f}", f"{it.monthly_amount():.2f}"])
    for it in expenses:
        w.writerow(["expense", it.name, it.frequency, f"{it.amount:.2f}", f"{it.monthly_amount():.2f}"])
    return buf.getvalue().encode("utf-8")

def export_pdf_bytes(incomes: List[LineItem],
                     expenses: List[LineItem],
                     inc_total_m: float,
                     exp_total_m: float,
                     balance_m: float,
                     chart_png_bytes: Optional[bytes]) -> bytes:
    """
    Try ReportLab for pretty tables + embedded chart; fallback to Matplotlib-only PDF.
    Returns PDF as bytes.
    """
    try:
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        out = io.BytesIO()
        doc = SimpleDocTemplate(out, pagesize=letter)
        styles = getSampleStyleSheet()
        elems = []

        month = date.today().strftime("%B %Y")
        elems.append(Paragraph(f"<b>Family Budget — {month}</b>", styles['Title']))
        elems.append(Spacer(1, 12))

        def make_table(title: str, items: List[LineItem]):
            data = [["Category", "Freq", "Entered", "Monthly", "% of total"]]
            total_m = sum(i.monthly_amount() for i in items) or 1.0
            for it in sorted(items, key=lambda z: z.monthly_amount(), reverse=True):
                pct = percent(it.monthly_amount(), total_m)
                data.append([it.name, it.frequency,
                             money(it.amount), money(it.monthly_amount()),
                             f"{pct:.1f}%"])
            total_line = money(sum(i.monthly_amount() for i in items))
            data.append(["TOTAL", "", "", total_line, "100.0%"])
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.grey),
                ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                ('ALIGN',(2,1),(-1,-1),'RIGHT'),
                ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                ('BOTTOMPADDING',(0,0),(-1,0),6),
                ('GRID',(0,0),(-1,-1),0.5,colors.black),
            ]))
            elems.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
            elems.append(t)
            elems.append(Spacer(1, 12))

        make_table("Income", incomes)
        make_table("Expenses", expenses)

        elems.append(Paragraph("<b>Summary</b>", styles['Heading2']))
        summary_data = [
            ["Total Monthly Income",   money(inc_total_m)],
            ["Total Monthly Expenses", money(exp_total_m)],
            ["Monthly Balance (Income - Expenses)", money(balance_m)],
        ]
        t = Table(summary_data, colWidths=[260, 180])
        t.setStyle(TableStyle([
            ('GRID',(0,0),(-1,-1),0.5,colors.black),
            ('ALIGN',(1,0),(1,-1),'RIGHT'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ]))
        elems.append(t)

        if chart_png_bytes:
            elems.append(PageBreak())
            elems.append(Paragraph("<b>Monthly Expenses Pie Chart</b>", styles['Heading2']))
            img_buf = io.BytesIO(chart_png_bytes)
            max_w = 7.0 * inch
            elems.append(RLImage(img_buf, width=max_w, height=max_w))  # square-ish

        doc.build(elems)
        return out.getvalue()

    except Exception:
        # Fallback: Matplotlib-only PDF
        out = io.BytesIO()
        with PdfPages(out) as pdf:
            month = date.today().strftime("%B %Y")
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            y = 0.95
            ax.text(0.5, y, f"Family Budget — {month}",
                    ha="center", va="top", fontsize=18, fontweight="bold"); y -= 0.06
            ax.text(0.05, y, f"Total Monthly Income : {money(inc_total_m)}", fontsize=12); y -= 0.04
            ax.text(0.05, y, f"Total Monthly Expenses: {money(exp_total_m)}", fontsize=12); y -= 0.04
            ax.text(0.05, y, f"Monthly Balance (Income - Expenses): {money(balance_m)}", fontsize=12); y -= 0.06

            # Income table (monospace)
            ax.text(0.05, y, "Income", fontsize=14, fontweight="bold"); y -= 0.03
            ax.text(0.05, y, f"{'Category':35} {'Freq':10} {'Entered':>12} {'Monthly':>12} {'%':>6}",
                    family="monospace", fontsize=9); y -= 0.02
            ax.axhline(y+0.01, xmin=0.05, xmax=0.95, linewidth=0.5)
            inc_total = sum(i.monthly_amount() for i in incomes) or 1.0
            for it in sorted(incomes, key=lambda z: z.monthly_amount(), reverse=True):
                line = f"{it.name:35} {it.frequency:10} {money(it.amount):>12} {money(it.monthly_amount()):>12} {percent(it.monthly_amount(), inc_total):5.1f}%"
                ax.text(0.05, y, line, family="monospace", fontsize=8); y -= 0.018
            y -= 0.02

            # Expenses table
            ax.text(0.05, y, "Expenses", fontsize=14, fontweight="bold"); y -= 0.03
            ax.text(0.05, y, f"{'Category':35} {'Freq':10} {'Entered':>12} {'Monthly':>12} {'%':>6}",
                    family="monospace", fontsize=9); y -= 0.02
            ax.axhline(y+0.01, xmin=0.05, xmax=0.95, linewidth=0.5)
            exp_total = sum(e.monthly_amount() for e in expenses) or 1.0
            for it in sorted(expenses, key=lambda z: z.monthly_amount(), reverse=True):
                line = f"{it.name:35} {it.frequency:10} {money(it.amount):>12} {money(it.monthly_amount()):>12} {percent(it.monthly_amount(), exp_total):5.1f}%"
                ax.text(0.05, y, line, family="monospace", fontsize=8); y -= 0.018

            pdf.savefig(fig); plt.close(fig)

            # Page 2: chart image if available
            if chart_png_bytes:
                img = plt.imread(io.BytesIO(chart_png_bytes))
                fig2, ax2 = plt.subplots(figsize=(8.5, 11))
                ax2.axis("off")
                ax2.set_title("Monthly Expenses Pie Chart")
                ax2.imshow(img)
                pdf.savefig(fig2); plt.close(fig2)

        return out.getvalue()

# ---------- Streamlit UI ----------

st.set_page_config(page_title="Budget Planner", page_icon="💸", layout="wide")

st.title("Family Budget Planner (Web)")
st.caption("CSV-first • Pie chart with smart labels • PDF report (chart embedded)")

with st.expander("Options", expanded=True):
    default_csv_name = "Family_budget.csv"
    csv_filename = st.text_input("CSV filename (for download name)", value=default_csv_name, help="Used as the suggested filename when you download the CSV.")
    pdf_filename = st.text_input("PDF filename (for download name)", value="budget_summary.pdf", help="Used as the suggested filename when you download the PDF.")
    show_chart = st.checkbox("Show chart below", value=True)
    embed_chart_in_pdf = st.checkbox("Embed pie chart in the PDF", value=True)

st.markdown("### Income")
if "income_df" not in st.session_state:
    st.session_state.income_df = pd.DataFrame({
        "name": ["Mom - income", "Dad - income"],
        "amount": [0.0, 0.0],
        "frequency": pd.Categorical(["monthly", "monthly"], categories=FREQ_CHOICES)
    })

income_df = st.data_editor(
    st.session_state.income_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("Name", required=False),
        "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=1.0, format="%.2f"),
        "frequency": st.column_config.SelectboxColumn("Frequency", options=FREQ_CHOICES, required=True),
    },
    key="income_editor"
)

st.markdown("### Expenses")
if "expense_df" not in st.session_state:
    expense_names = [
        "Mortgage","Pet insurance","Car insurance","Water bill","Electric bill",
        "therapy","childcare","meds","doctor","lessons",
        "kid1","kid2","kid3","kid4", # <-- fixed comma here
        "Gas (car)","Food","Unnecessary stuff","TV","Cell phones",
        "Internet","kids clothing","grown up clothing","gym/pool",
        "car/maintenance","emergency savings", "kids college plan",
        "yard/lawn maintenance","birthdays",
        "oh no, that was not planned!"
    ]
    n = len(expense_names)
    st.session_state.expense_df = pd.DataFrame({
        "name": expense_names,
        "amount": [0.0]*n,
        "frequency": pd.Categorical(["monthly"]*n, categories=FREQ_CHOICES)
    })

expense_df = st.data_editor(
    st.session_state.expense_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("Name", required=False),
        "amount": st.column_config.NumberColumn("Amount", min_value=0.0, step=1.0, format="%.2f"),
        "frequency": st.column_config.SelectboxColumn("Frequency", options=FREQ_CHOICES, required=True),
    },
    key="expense_editor"
)

# Convert to LineItems
def df_to_items(df: pd.DataFrame) -> List[LineItem]:
    items: List[LineItem] = []
    for _, r in df.fillna({"name":"", "amount":0.0, "frequency":"monthly"}).iterrows():
        name = str(r["name"]).strip() or "Item"
        try:
            amt = float(r["amount"]) if pd.notna(r["amount"]) else 0.0
        except Exception:
            amt = 0.0
        freq = str(r["frequency"]).strip().lower()
        if freq not in FREQ_TO_MONTHLY:
            freq = "monthly"
        items.append(LineItem(name=name, amount=amt, frequency=freq))
    return items

incomes = df_to_items(income_df)
expenses = df_to_items(expense_df)

# Totals
inc_total_m = round2(sum(i.monthly_amount() for i in incomes))
exp_total_m = round2(sum(e.monthly_amount() for e in expenses))
balance_m   = round2(inc_total_m - exp_total_m)

col1, col2, col3 = st.columns(3)
col1.metric("Total Monthly Income", money(inc_total_m))
col2.metric("Total Monthly Expenses", money(exp_total_m))
col3.metric("Monthly Balance", money(balance_m))

# --- Chart preview (optional) ---
chart_png_bytes: Optional[bytes] = None
fig = make_expense_pie(expenses)
if fig and show_chart:
    st.pyplot(fig, use_container_width=False)

# Always create chart bytes if embedding requested or if user wants to download image later
if fig and embed_chart_in_pdf:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    chart_png_bytes = buf.getvalue()
plt.close('all')  # free mem

st.divider()

# --- CSV download ---
csv_bytes = make_csv_bytes(incomes, expenses)
st.download_button(
    "⬇️ Download CSV",
    data=csv_bytes,
    file_name=csv_filename or "budget.csv",
    mime="text/csv",
    help="CSV contains both incomes and expenses with monthly amounts."
)

# --- PDF download (with optional embedded chart) ---
pdf_bytes = export_pdf_bytes(incomes, expenses, inc_total_m, exp_total_m, balance_m,
                             chart_png_bytes=chart_png_bytes if embed_chart_in_pdf else None)
st.download_button(
    "⬇️ Download PDF Report",
    data=pdf_bytes,
    file_name=pdf_filename or "budget_summary.pdf",
    mime="application/pdf",
    help="PDF includes income & expense tables, totals, and (optionally) the pie chart."
)

st.caption("Tip: Share this app by uploading it to Streamlit Community Cloud or running it locally with `streamlit run app.py`.")
