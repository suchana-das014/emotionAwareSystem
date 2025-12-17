import streamlit as st
from src.spam_model import predict_spam
from src.emotion_model import predict_emotion
from src.toxicity_model import predict_toxicity
from src.stress_index import compute_stress_index

st.set_page_config(page_title="Emotion-Aware NLP", layout="centered")

st.title("🧠 Emotion-Aware Mental Health System")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Flag to clear the input before widget creation when requested
if "should_clear" not in st.session_state:
    st.session_state.should_clear = False

if st.session_state.should_clear:
    # clear the keyed input before the widget is created
    st.session_state["input_text"] = ""
    st.session_state.should_clear = False



# Display chat history
if st.session_state.chat_history:
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.info(f"**You:** {msg['text']}")
        else:
            with st.expander("📊 Analysis Result"):
                st.write(msg['text'])

# Input for new message using a form so it resets automatically after submit
with st.form(key="input_form"):
    text = st.text_area("Enter SMS / Chat text", height=100, placeholder="Type your message here...", key="input_text")
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not text.strip():
        st.warning("Please enter some text")
    else:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "text": text})

        # Predict spam (returns 0 or 1, where 1 = spam)
        spam_pred = predict_spam(text)

        if spam_pred == 1:  # 1 means spam
            response = "🚫 Spam detected. Message ignored."
            st.session_state.chat_history.append({"role": "assistant", "text": response})
            st.error(response)
            # request clearing before next widget creation and rerun
            st.session_state.should_clear = True
            st.rerun()
        else:
            toxicity = predict_toxicity(text)
            # Pass toxicity to emotion predictor so it can override with "anger" for toxic messages
            emotion = predict_emotion(text, toxicity=toxicity)
            # compute_stress_index now accepts text, toxicity and emotion
            stress = compute_stress_index(text=text, toxicity=toxicity, emotion=emotion)

            # Format response with better spacing
            response = f"✅ Legitimate Message\n\n**Emotion:** {emotion}\n\n**Toxicity Level:** {toxicity}\n\n**Stress Index:** {stress}"

            if stress:
                response += "\n\n⚠️ This message may impact mental health"

            st.session_state.chat_history.append({"role": "assistant", "text": response})

            # Display results with better formatting
            st.success("✅ Legitimate Message")
            # Nicely spaced three-column layout with gaps
            st.write("")
            with st.container():
                cols = st.columns([1, 0.05, 1, 0.05, 1])
                with cols[0]:
                    st.metric("Emotion", emotion)
                # cols[1] is a spacer
                with cols[2]:
                    st.metric("Toxicity", "Yes" if str(toxicity) in {"1","True","true"} else "No")
                # cols[3] is a spacer
                with cols[4]:
                    st.metric("Stress Risk", "High" if stress == 1 else "Low")

            if stress:
                st.warning("⚠️ This message may impact mental health")
            # request clearing before next widget creation and rerun
            st.session_state.should_clear = True
            st.rerun()

# Button to clear chat history
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
