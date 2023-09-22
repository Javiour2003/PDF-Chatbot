from langchain.document_loaders import PyPDFLoader
from gpt4all import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model = GPT4All("orca-mini-3b.ggmlv3.q4_0.bin")

#creating context based on a pdf 
pdf = " " #Enter the location of the pdf here
pdf_reader = PyPDFLoader(pdf)
# text = ""
# for page in pdf_reader.pages:
#     text += page.extract_text()
documents = pdf_reader.load_and_split()

#split text for embeddings
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
chunks = text_splitter.split_documents(documents)

#create embeddings and store it in the vector stores
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks,embedding=embeddings)

#The pdf used is about SMA(Shape Memory Alloys)
question = "What is a shape memory alloy?"
context = db.similarity_search(question) #create context

#a prompt outline to get the best answer
prompt=f"""
Follow exactly those 3 steps:
1. Read the context below and aggregrate this data
Context : {context}
2. Answer the question using only this context
3. Show the source for your answers
User Question: {question} """

response = model.generate(prompt=prompt,
                        max_tokens=200, 
                        temp=0.4, 
                        top_k=40, 
                        top_p=0.4, 
                        repeat_penalty=1.18, 
                        repeat_last_n=64, 
                        n_batch=8, 
                        n_predict=None, 
                        streaming=False)

print(response)


#output
#Answer: A shape memory alloy (SMAs) is a metallic alloy that undergoes solid-to-solid 
# transformation which can exhibit large recoverable strains. 
# They have the special ability to remember their shape at any 
# temperature if deformed at low temperature, on heating they will 
# recover their shapes on thermal activation. This effect of recovering 
# the strain imparted to them at a low temperature, as
# a result of heating is called SME (shape memory effect). Examples of SMAs are Nitinol.










