import concurrent.futures
import os
import anthropic
import google.generativeai as genai
import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def fetch_openai(prompt):
  try:
    res = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    return "ChatGPT", res.choices[0].message.content
  except Exception as e:
    return "ChatGPT", f"Error: {e}"


def fetch_claude(prompt):
  try:
    res = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return "Claude", res.content[0].text
  except Exception as e:
    return "Claude", f"Error: {e}"


def fetch_gemini(prompt):
  try:
    model = genai.GenerativeModel("gemini-1.5-pro")
    res = model.generate_content(prompt)
    return "Gemini", res.text
  except Exception as e:
    return "Gemini", f"Error: {e}"


def judge_responses(prompt, answers):
  judge_prompt = f"""
    Sawal: {prompt}

    Answers:
    1. ChatGPT: {answers['ChatGPT']}
    2. Claude: {answers['Claude']}
    3. Gemini: {answers['Gemini']}

    Task:
    - Teeno answers ko evaluate karo.
    - Sabse accurate aur clear answer select karo ya teeno ka best synthesis banao.
    - Shuruat mein batao kaunsa answer sabse behtar tha aur kyun, fir final answer do.
    """
  res = openai_client.chat.completions.create(
      model="gpt-4o", messages=[{"role": "user", "content": judge_prompt}]
  )
  return res.choices[0].message.content


st.set_page_config(page_title="Nexus-AI", layout="wide")
st.title("Nexus-AI: Multi-Model Best Selector")

user_prompt = st.text_area("Apna sawal yahan likhein:", height=100)

if st.button("Generate & Pick Best Answer"):
  if not user_prompt.strip():
    st.warning("Pehle sawal likhein!")
  else:
    with st.spinner("Teeno APIs se answers fetch ho rahe hain..."):
      with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_openai, user_prompt),
            executor.submit(fetch_claude, user_prompt),
            executor.submit(fetch_gemini, user_prompt),
        ]
        results = dict([f.result() for f in futures])

    with st.spinner("Judge model best answer analyze kar raha hai..."):
      best_answer = judge_responses(user_prompt, results)

    st.subheader("Final Best Answer")
    st.success(best_answer)

    with st.expander("Individual Model Responses Dekhein"):
      col1, col2, col3 = st.columns(3)
      with col1:
        st.markdown("**ChatGPT (GPT-4o)**")
        st.write(results["ChatGPT"])
      with col2:
        st.markdown("**Claude (3.5 Sonnet)**")
        st.write(results["Claude"])
      with col3:
        st.markdown("**Gemini (1.5 Pro)**")
        st.write(results["Gemini"])
        
