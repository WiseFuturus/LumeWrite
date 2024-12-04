# flake8: noqa
from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Create a final answer to the given questions using the provided document excerpts (given in no particular order) as sources. ALWAYS include a "SOURCES" section in your answer citing only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not have enough information to answer the question and leave the SOURCES section empty. Use only the provided documents and do not attempt to fabricate an answer.

---------

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt's based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
SOURCES: 1-32
Content: While we're at it, let's make sure every American can get the health care they need. \n\nWe've already made historic investments in health care. \n\nWe've made it easier for Americans to get the care they need, when they need it. \n\nWe've made it easier for Americans to get the treatments they need, when they need them. \n\nWe've made it easier for Americans to get the medications they need, when they need them.
SOURCES: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat's why I'm calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
SOURCES: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer's, diabetes, and more.
SOURCES: 1-32

---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

QA_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)

generate_template = """

Generate detailed and coherent content based on the provided specific section title from the outline and the relevant retrieved document excerpts. 
The content should elaborate on the assigned title by synthesizing information from the provided excerpts. 
Ensure the writing is in a continuous, well-structured format without any subheadings or additional titles. 
Use only the information provided in the excerpts; do not fabricate any details. Focus solely on the given section and avoid referencing unrelated parts of the outline.

---------

Inputs:
TITLE:
{outline}

RETRIEVED DOCUMENTS:
{summaries}

---------

OUTPUT:
Write in plain text, paragraph style, providing detailed and engaging content relevant to the assigned section title.

"""

ARTICLE_PROMPT = PromptTemplate(
    template=generate_template, input_variables=["outline", "summaries"]
)


polish_template = """

您的任务是改进所提供的文章，需关注以下目标：

清晰和连贯：确保文章流畅，且各个部分自然过渡。必要时使用适当的连接词和短语。
吸引力和风格：优化语言，使其更加吸引人和专业。改变句子结构和词汇，避免单调并提高可读性。
精确性和一致性：确保所有论点都清晰表述并有证据支持。在全文中保持一致的语气和风格。
语法和句法：纠正任何语法、标点或句法错误。
简明性：消除冗余或过于冗长的部分，确保文章保持集中和简洁。
格式：将文本组织为定义明确的部分，并在必要时使用适当的标题或小标题。在能增强清晰度的地方使用项目符号或列表。
语言一致性：确保输出语言与所提供的文章语言一致。
需修改的文章： {article}

在改进文章后，请确保修订后的文本遵循这些指导原则，保留原始意义和目的。仅在能提升文章质量而不改变其核心内容的地方进行改进

"""

POLISH_PROMPT = PromptTemplate(
    template=polish_template, input_variables=["article"]
)
