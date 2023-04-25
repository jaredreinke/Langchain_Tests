from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


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
    temperature=1,
    request_timeout=120,
)

email_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

review_chain = load_qa_with_sources_chain(llm, chain_type="stuff")

subject = "shop our best sellers"

preview_text = "Trusted by thousands of others for their self-care needs"

marketing_voice = "energetic, excited, hyped, proud. ALL CAPS, YELLING, LOUD"

angle = "An excited and hyped product highlight campaign"

products = "Super Face Cream, prebiotic deodorant, hair-clay"

review_query = '''Scrape this html raw text output, and show me 5 positive reviews from the website.

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
'''

product_query = f'''You are SchwartzGPT. Use the principles from Breakthrough Advertising
by Eugene Schwartz to create creative, and highly converting email copy using this EXACT
subject line: "{subject}" and this EXACT preview text: "{preview_text}" and this angle {angle}.

create product blocks for each of the products listed here {products}

Use this format for each of the product blocks:

Product Name:
Header:
Body Text:

Product Name:
Header:
Body Text:

Product Name:
Header:
Body Text:

Product Name:
Header:
Body Text:

Etc for each product block listed here {products}

'''
product_blocks = email_chain.run(input_documents=docs, question=product_query)

print(product_blocks)

reviews = review_chain.run(input_documents=docs, question=review_query)

print(reviews)
