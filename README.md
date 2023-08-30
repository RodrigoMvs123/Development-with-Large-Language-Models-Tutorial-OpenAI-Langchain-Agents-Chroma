# Development-with-Large-Language-Models-Tutorial-OpenAI-Langchain-Agents-Chroma

https://www.youtube.com/watch?v=xZDB1naRUlk&t=2364s 

https://raw.githubusercontent.com/RodrigoMvs123/Development-with-Large-Language-Models-Tutorial-OpenAI-Langchain-Agents-Chroma/main/README.md

https://github.com/RodrigoMvs123/Development-with-Large-Language-Models-Tutorial-OpenAI-Langchain-Agents-Chroma/blame/main/README.md

https://github.com/pythonontheplane123/LLM_course_part_1 

What is a Large Language Model ( LLM ) ?
Deep learning techniques like massive neural network combined with huge amounts of data and then aligned to human values in an attempt to create a reasoning engine.

Examples: BERT, GPT3.5, GPT4, Llama



Under the hood

Steps: 
1 Choosing Architecture and tokens

How to make LLMs understand words? 
Tokenization: Converting words to numbers/tokens that the model can use mathematically.
Converting text to numbers and back.


Choosing the brain ( Model Architecture )





2 Training method

Next work/token predictic task:
Trained to predict these tokens by giving it billions of sentences /chunks and Answers
Data includes: Code, College Textbooks, Articles, lyrics, podcasts … etc

Multiple tokens?
Put the output of your predicted token back into inputs and so on until model outputs a stop sequence token


Step 1
Collect demonstration data, and train a supervised policy

Step 2
Collect comparison data, and train a reward model

Step 3 
Optimize a policy against the reward model using reinforcement learning



3 RLHF



4 Optional Fine-tinning



5 Inference, Prompting and Prompt Engineering 


OpenAI Playground

https://platform.openai.com/playground 

Playground

SYSTEM                                 USER Hi ! print out the first 10 digits of the fibonacci sequence  
                                                using Python
you are a programmer
                                                ASSISTANT Sure here is the code to print out the first 10 
                                                digits of the fibonacci sequence in Python
```python
def fibonacci(n):
    fib = [0, 1]
    for i in range [2, n]:
       fib.append(fib[i+1] + fib[i-2])
    return fib

   fib_sequence = fibonacci(10)
    for num in fib_sequence:
       print(num)
``` 

0, 1, 1, 2, 3, 5, 8, 13, 21, 34.

API Keys
Create new secret key 
…

https://colab.research.google.com/drive/1gi2yDvvhUwLT7c8ZEXz6Yja3cG5P2owP?usp=sharing 


Chainlit.io
https://docs.chainlit.io/overview 

Visual Studio Code
EXPLORER
OPEN EDITORS 
main.py
https://github.com/pythonontheplane123/LLM_course_part_1/blob/main/main.py 

main.py
import chainlit as cl
import openai
import os


def get_gpt_output(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role":"system","content":"you are an assistant that is obsessed with potatoes and will never stop talking about them"},
            {"role":"user","content": user_message}
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response

@cl.on_message
async def main(message : str):
    await cl.Message(content = f"{get_gpt_output(message)['choices'][0]['message']['content']}",).send()

Visual Studio Code
Terminal
pip install chainlit
pip install openai
chainlit run main.py -w ( localhost:8000 )

LangChain
https://www.langchain.com/ 

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py

langchain_integration.py
import chainlit as cl
import openai
import os
from langchain import PromptTemplate, OpenAI, LLMChain


template = """Question: {question}

Answer: Let's think step by step."""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables = ["question"])
    llm_chain = LLMChain(prompt = prompt,llm=OpenAI(temperature=0,streaming=True),verbose=True)

    cl.user_session.set("llm_chain",llm_chain)

@cl.on_message
async def main(message : str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()

Visual Studio Code
Terminal
pip install -U langchain
chainlit run langchain_integration.py -w ( localhost:8000 )

Vector DBs and Embeddings 

Embeddings
Recommendation system
Search Engines
Generative AI
Memory for LLMs
Context window expansion
Agents like AutoGPT

Vector Databases
Fast retrieval of the relevant context from embeddings
Convenient storage embeddings
Context length augmentation 


Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py

chroma_db_basics.py
import chromadb

chroma_client = chromadb.Client()

collections = chroma_client.create_collection(name = "my_collection")

collections.add(
    documents = ["my name is akshath","my name is not akshath"],
    metadatas = [{"source":"my_source"},{"source":"my_source"}],
    ids = ["id1","id2"]
)

results = collections.query(
    query_texts = ["what is my name"],
    n_results = 1
)

print(results)

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
document._qa.py

document._qa.py
import os

#pip install pypdf
#export HNSWLIB_NO_NATIVE = 1

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl
from chainlit.types import AskFileResponse




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""


def process_file(file: AskFileResponse):
    import tempfile

    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as tempfile:
        tempfile.write(file.content)
        loader = Loader(tempfile.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file

    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


@cl.on_chat_start
async def start():
    # Sending an image with the local file path
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # No async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
    )

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])

    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []

    # Get the documents from the user session
    docs = cl.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()

Visual Studio Code
Terminal
pip install chroma
pip install chromadb
export HNSWLIB_NO_NATIVE = 1
chainlit run document_qa.py 

Web Browsing + Agents
Project #3
Project #4

Why Browse the Web ?
There is a knowledge cutoff
Databases vs Reasoning Engines 
Reduce bias incurred while training

Agents 
The problem
Find the answer to “what is RLHF” ?

Resources Langchain Gives us:
https://arxiv.org/ 

https://python.langchain.com/docs/integrations/providers/arxiv 

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py

internet_browsing_Arxiv_Naive.py
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os


#If the parser is erroring out, remember to set temperature to a higher value!!!

#pip install arxiv

llm = ChatOpenAI(temperature=0.3)
tools = load_tools(
    ["arxiv"]
)

agent_chain = initialize_agent(
    tools,
    llm,
    max_iterations=5,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True, ### IMPORTANT
)

agent_chain.run(
    "what is RLHF?",
)

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py
internet_browsing_Arxiv_chainlit.py

internet_browsing_Arxiv_chainlit.py
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os



@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    tools = load_tools(
        ["arxiv"]
    )

    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  ### IMPORTANT
    )

    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])

Visual Studio Code
Terminal
chainlit run internet_browsing_Arxiv_chainlit.py -w ( localhost:8000 )

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py
internet_browsing_Arxiv_chainlit.py
Python_replit.py

Python_replit.py
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
import os


agent_executor = create_python_agent(
    llm=OpenAI(temperature=0.5, max_tokens=2000),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent_executor.run("What is the 10th fibonacci number?")

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py
internet_browsing_Arxiv_chainlit.py
Python_replit.py
Youtube_search.py

Youtube_search.py
#! pip install youtube_search

from langchain.tools import YouTubeSearchTool
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import os



tool = YouTubeSearchTool()

tools = [
    Tool(
        name="Search",
        func=tool.run,
        description="useful for when you need to give links to youtube videos. Remember to put https://youtube.com/ in front of every link to complete it",
    )
]


agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run('Whats a joe rogan video on an interesting topic')

Visual Studio Code
Terminal
pip install youtube_search

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py
internet_browsing_Arxiv_chainlit.py
Python_replit.py
Youtube_search.py
CLI_GPT.py

CLI_GPT.py
from langchain.tools import ShellTool

shell_tool = ShellTool()

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os


llm = ChatOpenAI(temperature=0)

shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")
agent = initialize_agent(
    [shell_tool], llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "create a text file called empty and inside it, add code that trains a basic convolutional neural network for 4 epochs"
)

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py
internet_browsing_Arxiv_chainlit.py
Python_replit.py
Youtube_search.py
CLI_GPT.py
Combination_of_both.py

Combination_of_both.py
#https://python.langchain.com/docs/modules/agents/tools/multi_input_tool

from langchain import OpenAI, LLMMathChain, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
import os
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os
from langchain.tools import ShellTool




shell_tool = ShellTool()


@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0)

    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")

    agent = initialize_agent(
        [shell_tool],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors = True
    )
    cl.user_session.set("agent", agent)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    await cl.make_async(agent.run)(message, callbacks=[cb])

Visual Studio Code
Terminal
chainlit run Combination_f_both.py -w ( localhost:8000 )

Visual Studio Code
EXPLORER
OPEN EDITORS
main.py
langchain_integration.py
chroma_db_basics.py
internet_browsing_Arxiv_Naive.py
internet_browsing_Arxiv_chainlit.py
Python_replit.py
Youtube_search.py
CLI_GPT.py
Combination_of_both.py
Custom_tools.py

Custom_tools.py
import os

os.environ["LANGCHAIN_TRACING"] = "true"

from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

def multiplier(a, b):
    return a / b


def parsing_multiplier(string):
    a, b = string.split(",")
    return multiplier(int(a), int(b))

llm = OpenAI(temperature=0)
tools = [
    Tool(
        name="Multiplier",
        func=parsing_multiplier,
        description="useful for when you need to multiply two numbers together. The input to this tool should be a comma separated list of numbers of length two, representing the two numbers you want to multiply together. For example, `1,2` would be the input if you wanted to multiply 1 by 2.",
    )
]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run("3 times four?")



