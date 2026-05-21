import streamlit as st
import pandas as pd
import yaml
from metrics import (
    semantic_similarity,
    lexical_diversity,
    avg_sentence_length,
    information_loss
)

# Load config
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

app_cfg     = config["app"]
models_cfg  = config["models"]
weights_cfg = config["scoring"]["weights"]
export_cfg  = config["export"]

st.set_page_config(
    page_title=app_cfg["title"],
    layout=app_cfg["layout"]
)

st.title(app_cfg["title"])
st.write(app_cfg["description"])

# Original text input
original_text = st.text_area(
    "Original Kannada Text",
    height=150
)

# Dynamically generate summary inputs from config
summaries = {}
for key, model in models_cfg.items():
    summaries[model["label"]] = st.text_area(
        f"{model['label']} Summary ({model['name']})",
        height=100
    )

# Evaluation button
if st.button("Evaluate Models"):

    if not original_text or not all(summaries.values()):
        st.warning("Please fill all text fields.")

    else:
        data = []

        for label, summary in summaries.items():
            sem              = semantic_similarity(original_text, summary)
            lex              = lexical_diversity(summary)
            sent             = avg_sentence_length(summary)
            missing, added   = information_loss(original_text, summary)

            data.append({
                "Model":               label,
                "Semantic Similarity": round(sem, 4),
                "Lexical Diversity":   round(lex, 4),
                "Avg Sentence Length": round(sent, 2),
                "Missing Info":        missing,
                "Added Info":          added
            })

        df = pd.DataFrame(data)

        # Weighted final score from config
        df["Final Score"] = (
            weights_cfg["semantic_similarity"] * df["Semantic Similarity"] +
            weights_cfg["lexical_diversity"]   * df["Lexical Diversity"] +
            weights_cfg["avg_sentence_length"] * (
                df["Avg Sentence Length"] / df["Avg Sentence Length"].max()
            )
        ).round(4)

        # Ranking
        df = df.sort_values("Final Score", ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"

        best_model = df.iloc[0]["Model"]

        # Results
        st.subheader("Model Evaluation Results")
        st.dataframe(df, use_container_width=True)
        st.success(f"🏆 Best Overall Performing Model: {best_model}")

        # Charts
        st.subheader("Semantic Similarity Comparison")
        st.bar_chart(df.set_index("Model")["Semantic Similarity"])

        st.subheader("Lexical Diversity Comparison")
        st.bar_chart(df.set_index("Model")["Lexical Diversity"])

        st.subheader("Information Coverage Analysis")
        st.bar_chart(df.set_index("Model")[["Missing Info", "Added Info"]])

        st.subheader("Final Score Comparison")
        st.bar_chart(df.set_index("Model")["Final Score"])

        # Download CSV
        st.download_button(
            "Download Results as CSV",
            df.to_csv(index=True),
            export_cfg["default_filename"],
            mime="text/csv"
        )

# Model disclosure — dynamically built from config
model_lines = "\n".join(
    f"     {v['label']} → {v['name']}" for v in models_cfg.values()
)
st.caption(f"Model Mapping:\n{model_lines}")
