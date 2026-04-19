import os 
import re
import streamlit as st
import tempfile
from pathlib import Path
from src.online.query import run_query
from src.offline.offline import run_offline_pipeline

def extract_image_urls(text: str):
    compact_text = text

    compact_text = compact_text.replace("http:// ", "http://")
    compact_text = compact_text.replace("https:// ", "https://")

    compact_text = re.sub(r'\s*\.\s*', '.', compact_text)
    compact_text = re.sub(r'\s*/\s*', '/', compact_text)
    compact_text = re.sub(r'\s*:\s*', ':', compact_text)

    pattern = r'https?://[^\s<>"\']+?\.(?:jpg|jpeg|png|webp)'
    return re.findall(pattern, compact_text, flags=re.IGNORECASE)

st.set_page_config(page_title="Financial RAG Demo", layout="wide")

st.title("Financial RAG Demo")
st.write("这是一个简单的 RAG 演示页面。")

st.header("1. Offline Pipeline")

uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

chunk_size = st.number_input("chunk size", min_value=50, max_value=2000, value=400, step=50)
overlap = st.number_input("overlap", min_value=0, max_value=500, value=100, step=10)

run_offline = st.button("运行 Offline Pipeline")

if run_offline:
    if uploaded_file is None:
        st.error("请先上传 PDF 文件。")
    else:
        try:
            save_dir = Path("data/raw/demo_uploads")
            save_dir.mkdir(parents=True, exist_ok=True)

            saved_pdf_path = save_dir / uploaded_file.name
            with open(saved_pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            st.info("Offline pipeline 正在运行，请稍等...")

            result = run_offline_pipeline(
                pdf_path=str(saved_pdf_path),
                chunk_size=int(chunk_size),
                overlap=int(overlap),
                drop_old=False,
            )

            st.success("Offline pipeline 运行完成。")
            st.json(result)

        except Exception as e:
            st.error(f"运行失败: {e}")

st.header("2. Online Query")

query = st.text_input("输入你的问题")
top_k = st.number_input("top_k", min_value=1, max_value=10, value=3, step=1)
run_query_btn = st.button("开始提问")

if run_query_btn:
    if not query.strip():
        st.error("请先输入问题。")
    else:
        try:
            st.info("Online query 正在运行，请稍等...")

            query_result = run_query(
                query=query,
                top_k=int(top_k),
            )

            st.success("Online query 运行完成。")

            st.subheader("Answer")
            st.write(query_result["answer"])

            st.subheader("Citations")
            for c in query_result["citations"]:
                st.markdown(f"**{c['citation_mark']} {c['source']} / chunk_{c['chunk_id']}**")
                st.code(repr(c["text"][:1000]))
                st.write(c["text"][:500])
                
                image_urls = extract_image_urls(c["text"])
                st.write("debug image_urls:", image_urls)
                
                for image_url in image_urls:
                    st.image(image_url, caption=image_url)

                st.divider()

        except Exception as e:
            st.error(f"提问失败: {e}")

st.header("3. Output")
st.info("这里之后会显示 offline 运行状态和 online 问答结果。")