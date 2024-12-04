import streamlit as st

def display_file_upload():
    return st.file_uploader(
        "支持上传 pdf、docx 或 txt 文件",
        type=["pdf", "docx", "txt", "md"],
        help="暂不支持扫描文档～!",
        accept_multiple_files=True
    )

def display_sources(result):
    with st.expander("检索源"):
        for source in result.get("sources", []):
            st.markdown(
                f"<p style='font-size:12px;'><b>{source['metadata']['name']}</b></p>",
                unsafe_allow_html=True
            )
            st.markdown(source["page_content"])
