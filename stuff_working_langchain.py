from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

urls = [
    "https://jackhenry.co/",
    "https://jackhenry.co/pages/about",
    "https://jackhenry.co/products/super-face-cream",
    "https://jackhenry.co/collections/best-sellers/products/deodorant",
    "https://jackhenry.co/collections/best-sellers/products/hair-clay",
]


loader = SeleniumURLLoader(urls=urls)

html = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

docs = text_splitter.split_documents(html)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    request_timeout=120,
)

chain = load_qa_chain(llm, chain_type="stuff")

query = """Scrape this html raw text output, and show me 5 positive reviews from the website.

Only use the factual reviews from the website to answer the question.

If you feel like you don't have enough information to answer the question, say "I don't know".

Do not make up reviews, only pull from the html text I give you. 

Use this format for each review

Product Name:
Number of Stars:
Review Header:
"Review text"
 - Reviewer Name

Product Name:
Number of Stars:
Review Header:
"Review text"
 - Reviewer Name

Product Name:
Number of Stars:
Review Header:
"Review text"
 - Reviewer Name

Product Name:
Number of Stars:
Review Header:
"Review text"
 - Reviewer Name

Product Name:
Number of Stars:
Review Header:
"Review text"
 - Reviewer Name
"""

response = chain.run(input_documents=docs, question=query)

print(response)