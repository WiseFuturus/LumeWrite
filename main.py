import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from ui.helpers import is_query_valid,is_file_valid,display_file_read_error

from core.parsing import read_file
from core.chunking import chunk_file
from core.qa import query_answer
from core.outline import generate_outline_content
from core.faiss import FAISSStore
from core.article import generate_section,polish_content

load_dotenv()

LLM  = ChatZhipuAI(
    api_key=os.getenv('ZhipuAI_API_KEY'),
    model="glm-4-plus",
)

EMBEDDINGS = ZhipuAIEmbeddings(model="embedding-2",api_key=os.getenv('ZhipuAI_API_KEY'))

st.set_page_config(page_title="📖LumeWriter", page_icon="📖", layout="wide")
st.header("📖LumeWriter")

uploaded_files = st.file_uploader(
    "支持上传 pdf、docx 或 txt 文件",
    type=["pdf", "docx", "txt", "md"],
    help="暂不支持扫描文档～!",
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

files = []
for uploaded_file in uploaded_files:
    try:
        file = read_file(uploaded_file)
        files.append(file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)

chunked_files = [chunk_file(file, chunk_size=300, chunk_overlap=0) for file in files]

if not all(is_file_valid(file) for file in files):
    st.stop()

with st.spinner("向量索引构建中... 请稍等⏳"):
    if "vectorstore_initialized" not in st.session_state:
        st.session_state.vectorstore,st.session_state.chunked_docs = FAISSStore(embeddings=EMBEDDINGS).initialize_vectorstore(chunked_files)
        st.session_state.vectorstore_initialized = True
        st.success("向量索引构建完成！")

vectorstore = st.session_state.vectorstore    
chunked_docs = st.session_state.chunked_docs

with st.container():
    with st.form(key="qa_form"):
        query = st.text_area("请输入你的问题")
        submit = st.form_submit_button("提交")

        if submit:
            if not is_query_valid(query):
                st.warning("输入问题无效，请检查后重新输入！")
                st.stop()

            result = query_answer(query,vectorstore,LLM,EMBEDDINGS)

            with st.expander("检索源"):
                for source in result.sources:
                    st.markdown(
                        f"<p style='font-size:12px;'><b style='color:#007BFF;'>{source.metadata['name']}</b></p>",
                        unsafe_allow_html=True
                    )
                    st.markdown(source.page_content)
                    st.markdown(
                        f"<p style='font-size:12px;'>{source.metadata['source']}</p>",
                        unsafe_allow_html=True
                    )
                    st.markdown("---")

            st.markdown("#### 回答")
            st.markdown(result.answer)

with st.container():

    generate_outline = st.button("生成写作大纲")
    if generate_outline:
        outline_content = generate_outline_content(chunked_docs, LLM)
        if 'editable_outline' not in st.session_state:
            st.session_state['editable_outline'] = outline_content

    def display_editable_outline():
        outline = st.session_state['editable_outline']
        outline_title = st.text_input("大纲标题", outline['title'], key="outline_title")
        outline['title'] = outline_title

        for i, section in enumerate(outline['sections']):
            with st.expander(f"第 {i + 1} 章 : {section['heading']}"):
                section_heading = st.text_input(
                    f"第 {i + 1} 章", section['heading'], key=f"section_heading_{i}"
                )
                subsections = []
                for j, subsection in enumerate(section['subsections']):
                    updated_subsection = st.text_input(
                        f"{i + 1}.{j + 1}", subsection, key=f"subsection_{i}_{j}"
                    )
                    subsections.append(updated_subsection)

                outline['sections'][i]['heading'] = section_heading
                outline['sections'][i]['subsections'] = subsections

        st.session_state['editable_outline'] = outline

    if 'editable_outline' in st.session_state:
        display_editable_outline()

    if st.button("保存更新"):
        st.success("大纲已更新！")

    if 'editable_outline' in st.session_state:
        st.write("更新后的大纲 JSON 数据：")
        st.json(st.session_state['editable_outline'])

with st.container():

    if st.session_state.get("editable_outline") and st.button("继续生成全文"):
        if 'structured_content' not in st.session_state:
            st.session_state['structured_content'] = {}
        outline = st.session_state['editable_outline']
        for index,row in enumerate(outline['sections']):
            row_heading = row['heading']
            for sub_index,subsection in enumerate(row['subsections']):
                query = f"{row_heading}-{subsection}"
                subsection_generate_content = generate_section(query, vectorstore, LLM,EMBEDDINGS)
                st.session_state['structured_content'][query] = subsection_generate_content['content']
                st.markdown(f"第 {index + 1} 章 {row_heading} - {subsection}")
                st.markdown(subsection_generate_content["content"])

                for source in subsection_generate_content.get("sources", []):
                    with st.expander(f"📄 来源: {source['metadata']['name']}"):
                        st.markdown(source["page_content"])
                        st.markdown(
                            f"<p style='font-size:12px;'>来源: {source['metadata']['source']}</p>",
                            unsafe_allow_html=True
                        )

                st.markdown("---")

with st.container():
    st.markdown("---")
    if 'structured_content' in st.session_state and "editable_outline" in st.session_state:
        editable_outline = st.session_state['editable_outline']
        structured_content = st.session_state['structured_content']
        if st.button("一键润色全文"):
            st.write("润色后的全文如下：")
            full_article = f"文章标题: {editable_outline['title']}\n\n"
            for index,row in enumerate(editable_outline["sections"]):
                row_heading = row['heading']
                st.markdown(f"### 正在润色第 {index + 1} 章: {row_heading}...")
                chapter_content = f"第 {index + 1} 章: {row_heading}\n\n"
                for sub_index, subsection in enumerate(row['subsections']):
                    query = f"{row_heading}-{subsection}"
                    content = structured_content.get(query, "")
                    chapter_content += f"{sub_index + 1}. {subsection}\n{content}\n\n"
                polished_chapter = polish_content(chapter_content, LLM)
                st.markdown(f"#### 第 {index + 1} 章润色完成：")
                st.markdown(polished_chapter)
                st.markdown("---")

                full_article += polished_chapter + "\n\n"

            st.write("### 全文润色结果：")
            st.markdown(full_article)    
