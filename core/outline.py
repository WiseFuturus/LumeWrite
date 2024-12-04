from typing import List
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatZhipuAI

class Section(BaseModel):
    heading: str = Field(description="部分标题")
    subsections: List[str] = Field(description="子部分列表")

class ResponseModel(BaseModel):
    title: str = Field(description="文章的标题")
    sections: List[Section] = Field(description="文章的各部分及其子部分")

def generate_outline_content(documents, model):

    prompt_template = """
        根据以下文章内容生成一个详细的写作大纲，要求包括标题和每个部分的子部分，并严格以 JSON 格式返回，不要包含其他说明文字：
        文章内容:
        {content}

        大纲的格式如下：
        {{
            "title": "文章的标题",
            "sections": [
                {{"heading": "部分标题", "subsections": ["子部分1", "子部分2"]}}
            ]
        }}
    """

    structured_chat = model.with_structured_output(ResponseModel)
    content = "\n\n".join([i.page_content for i in documents])
    prompt = prompt_template.format(content=content)
    response = structured_chat.invoke(prompt)

    result = {
        "title": response.title,
        "sections": [{"heading": section.heading, "subsections": section.subsections} for section in response.sections]
    }

    return result

