import streamlit as st
import pandas as pd

from metrics import (
    semantic_similarity,
    lexical_diversity,
    avg_sentence_length,
    information_loss
)

st.set_page_config(
    page_title="KanEval",
    layout="centered"
)

st.title("KanEval: Kannada LLM Evaluation Framework")

st.write(
    """
    Comparative evaluation of multiple Large Language Models (LLMs)
    for Kannada text summarization using semantic, lexical,
    structural, and information coverage metrics.
    """
)

# Original text input
original_text = st.text_area(
    "Original Kannada Text",
    height=150
)

# Model summary inputs
model_a = st.text_area(
    "Model A Summary",
    height=100
)

model_b = st.text_area(
    "Model B Summary",
    height=100
)

model_c = st.text_area(
    "Model C Summary",
    height=100
)

# Evaluation button
if st.button("Evaluate Models"):

    if not original_text or not model_a or not model_b or not model_c:
        st.warning("Please fill all text fields.")

    else:

        models = {
            "Model A": model_a,
            "Model B": model_b,
            "Model C": model_c
        }

        data = []

        for name, summary in models.items():
            sem = semantic_similarity(original_text, summary)

            lex = lexical_diversity(summary)

            sent = avg_sentence_length(summary)

            missing, added = information_loss(
                original_text,
                summary
            )

            data.append({
                "Model": name,
                "Semantic Similarity": round(sem, 4),
                "Lexical Diversity": round(lex, 4),
                "Avg Sentence Length": round(sent, 2),
                "Missing Info": missing,
                "Added Info": added
            })

        # Create dataframe
        df = pd.DataFrame(data)

        # Final weighted score
        df["Final Score"] = (
                0.6 * df["Semantic Similarity"] +
                0.25 * df["Lexical Diversity"] +
                0.15 * (
                        df["Avg Sentence Length"] /
                        df["Avg Sentence Length"].max()
                )
        )

        df["Final Score"] = df["Final Score"].round(4)

        # Ranking
        df = df.sort_values(
            "Final Score",
            ascending=False
        ).reset_index(drop=True)

        df.index = df.index + 1
        df.index.name = "Rank"

        best_model = df.iloc[0]["Model"]

        # Results section
        st.subheader("Model Evaluation Results")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.success(
            f"🏆 Best Overall Performing Model: {best_model}"
        )

        # Charts
        st.subheader("Semantic Similarity Comparison")
        st.bar_chart(
            df.set_index("Model")["Semantic Similarity"]
        )

        st.subheader("Lexical Diversity Comparison")
        st.bar_chart(
            df.set_index("Model")["Lexical Diversity"]
        )

        st.subheader("Information Coverage Analysis")
        st.bar_chart(
            df.set_index("Model")[
                ["Missing Info", "Added Info"]
            ]
        )

        st.subheader("Final Score Comparison")
        st.bar_chart(
            df.set_index("Model")["Final Score"]
        )

        # Download CSV
        st.download_button(
            "Download Results as CSV",
            df.to_csv(index=True),
            "kaneval_results.csv",
            mime="text/csv"
        )

# Model disclosure
st.caption(
    """
    Model Mapping:

    Model A → GPT-4

    Model B → Gemini

    Model C → DeepSeek
    """
)
