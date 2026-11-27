                    comparison_probability = 1.0 - output
                    prediction = (
                        "PCO"
                        if pcos_probability >= comparison_probability
                        else "Comparison Class"
                    )

                    advisory = None
                    advisory_error = None
                    try:
                        advisory = generate_ultrasound_advisory(
                            prediction,
                            pcos_probability,
                            comparison_probability
                        )
                    except Exception as exc:
                        advisory_error = str(exc)

                    st.session_state["ultrasound_result"] = {
                        "prediction": prediction,
                        "pcos_probability": pcos_probability,
                        "comparison_probability": comparison_probability,
                        "advisory": advisory,
                        "advisory_error": advisory_error,
                        "filename": uploaded_file.name,
                        "image_size": image.size
                    }
            except Exception as exc:
                st.error(f"Ultrasound prediction error: {exc}")
                st.session_state.pop("ultrasound_result", None)

    ultrasound_result = st.session_state.get("ultrasound_result")
    if ultrasound_result:
        current_filename = (
            uploaded_file.name if uploaded_file is not None else None
        )
        if current_filename == ultrasound_result["filename"]:
            prediction = ultrasound_result["prediction"]
            st.markdown(
                f"<div class='prediction-title'>Predicted Class: "
                f"{prediction}</div>",
                unsafe_allow_html=True
            )

            confidence = max(
                ultrasound_result["pcos_probability"],
                ultrasound_result["comparison_probability"]
            )
            st.metric("Prediction Confidence", f"{confidence * 100:.2f}%")

            probability_df = pd.DataFrame({
                "Class": ["PCO", "Comparison Class"],
                "Probability (%)": [
                    round(ultrasound_result["pcos_probability"] * 100, 2),
                    round(ultrasound_result["comparison_probability"] * 100, 2)
                ]
            })

            with st.expander("📊 Probability Analysis", expanded=True):
                st.bar_chart(
                    probability_df.set_index("Class")["Probability (%)"],
                    color="#9B7BC4"
                )
                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

            with st.expander("💡 Ultrasound AI Advisory", expanded=False):
                if ultrasound_result["advisory"]:
                    st.markdown(ultrasound_result["advisory"])
                    st.info(ADVISORY_DISCLAIMER)
                else:
                    st.warning(
                        "The ultrasound advisory could not be generated. "
                        f"{ultrasound_result['advisory_error'] or ''}"
                    )

            with st.expander("🖼️ Image Details", expanded=False):
                st.write(f"**File name:** {ultrasound_result['filename']}")
                st.write(
                    f"**Original image dimensions:** "
                    f"{ultrasound_result['image_size'][0]} × "
                    f"{ultrasound_result['image_size'][1]} pixels"
                )
                st.caption(
                    "The CNN classifies the uploaded image as PCO or "
                    "Comparison Class only. It does not predict the four "
                    "clinical PCOS subtypes."
                )
        else:
            st.info("Upload and analyze an image to view its prediction.")


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    "<div class='app-footer'>PCOS Insight · Research and educational use only · "
    "Not a medical diagnostic tool</div>",
    unsafe_allow_html=True
)
