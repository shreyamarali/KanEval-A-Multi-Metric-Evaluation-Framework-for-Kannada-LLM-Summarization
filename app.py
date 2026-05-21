import streamlit as st
import pandas as pd

from metrics import (
    semantic_similarity,
    lexical_diversity,
    avg_sentence_length,
    information_loss
)

st.set_page_config(page_title="Kannada LLM Evaluation", layout="centered")

st.title("Kannada LLM NLP Comparison")
st.write(
    "Advanced evaluation of Kannada GPT, Gemini, and YesChat "
    "using semantic, lexical, structural, and faithfulness metrics."
)



original_text = st.text_area("Original Kannada Text", height=150)

kgpt = st.text_area("Kannada GPT Summary", height=100)
gemini = st.text_area("Gemini Summary", height=100)
yeschat = st.text_area("YesChat Summary", height=100)



if st.button("Evaluate Models"):
    if not original_text or not kgpt or not gemini or not yeschat:
        st.warning("Please fill all text fields.")
    else:
        models = {
            "Kannada GPT": kgpt,
            "Gemini": gemini,
            "YesChat": yeschat
        }

        data = []

        for name, summary in models.items():
            sem = semantic_similarity(original_text, summary)
            lex = lexical_diversity(summary)
            sent = avg_sentence_length(summary)
            missing, added = information_loss(original_text, summary)

            data.append({
                "Model": name,
                "Semantic Similarity": sem,
                "Lexical Diversity": lex,
                "Avg Sentence Length": sent,
                "Missing Info (count)": missing,
                "Added Info (count)": added
            })

        df = pd.DataFrame(data)



        df["Final Score"] = (
            0.6 * df["Semantic Similarity"] +
            0.25 * df["Lexical Diversity"] +
            0.15 * (df["Avg Sentence Length"] / df["Avg Sentence Length"].max())
        )

        best_model = df.loc[df["Final Score"].idxmax(), "Model"]



        st.subheader("Model Evaluation Results")
        st.dataframe(
            df.sort_values("Final Score", ascending=False),
            use_container_width=True
        )

        st.success(f"🏆 Best Overall Model: {best_model}")



        st.subheader("Semantic Similarity Comparison")
        st.bar_chart(df.set_index("Model")["Semantic Similarity"])

        st.subheader("Lexical Diversity Comparison")
        st.bar_chart(df.set_index("Model")["Lexical Diversity"])

        st.subheader("Hallucination Analysis")
        st.bar_chart(
            df.set_index("Model")[["Missing Info (count)", "Added Info (count)"]]
        )



        st.download_button(
            "Download Results as CSV",
            df.to_csv(index=False),
            "kannada_llm_evaluation_results.csv"
        )
