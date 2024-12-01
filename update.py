import os
import torch
import streamlit as st
from pinecone import Pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from PIL import Image

 


 



# Streamlit app customization using CSS
st.markdown("""
    <style>
        /* Set the background color */
        .main {
            background-color: gray;
        }
        /* Customize button color */
        div.stButton > button {
            background-color: brown;
            color: white;
            border-radius: 5px;
            border: 1px solid black;
        }
        .chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
     }
       .chat-message.user {
         background-color: #2b313e
         }
       .chat-message.bot {
        background-color: #475063
          }
       .chat-message .avatar {
         width: 20%;
         }
        .chat-message .avatar img {
          max-width: 78px;
          max-height: 78px;
          border-radius: 50%;
          object-fit: cover;
           }
        .chat-message .message {
         width: 80%;
          padding: 0 1.5rem;
         color: #fff;
         }

        [data-testid="stSidebar"] .stSelectbox {
        background-color: skyblue !important;
        border-radius: 5px;
        padding: 5px;
    }

        .stTextInput>div>div>input {
            border: 2px solid #4CAF50; /* Green border */
            border-radius: 10px; /* Rounded corners */
            padding: 10px; /* Padding inside the input */
            font-size: 18px; /* Larger font size */
            width: 100%; /* Full width */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* Shadow for a subtle effect */
        }
        .stTextInput label {
            font-size: 20px; /* Label font size */
            font-weight: bold; /* Bold label */
            color: #333; /* Dark label color */

        .header-container {
            display: flex;
            align-items: center;
        }
        .header-container img {
            height: 60px; /* Adjust the logo height */
            margin-right: 15px; /* Space between logo and title */
        }
        .header-container h1 {
            font-size: 28px;
            color: #333;
            margin: 0;
        }


        /* Set base theme and background color */
        body {
            background-color: #d2c5c5;
            color: #000; /* Set text color for better contrast on light background */
        }

        /* Customize the Streamlit container */
        .stApp {
            background-color: #d2c5c5;
        }

        /* Customize sidebar */
        .sidebar .sidebar-content {
            background-color: #d2c5c5;
        }

        #logo desgin
        .header-container {
    display: flex;
    align-items: center;
    gap: 20px; /* Increased gap between the logo and text */
    padding: 20px; /* Added padding for a cleaner layout */
}

.header-container img {
    width: 300px; /* Increased logo width */
    height: auto; /* Maintain aspect ratio for the logo */
    border-radius: 10px; /* Added slight rounding for a professional touch */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow for emphasis */
}

.header-container h1 {
    font-size: 32px; /* Increased font size for better visibility */
    margin: 0;
    padding: 0;
    color: #222; /* Slightly darker text for contrast */
    font-family: 'Arial', sans-serif; /* Updated font for a modern look */
    line-height: 1.2; /* Adjusted line height for readability */
}

         


    </style>
""", unsafe_allow_html=True)

 
css = """
    .header-container {
    display: flex;
    align-items: center;
    gap: 20px; /* Increased gap between the logo and text */
    padding: 20px; /* Added padding for a cleaner layout */
}

    .header-container img {
    width: 300px; /* Increased logo width */
    height: auto; /* Maintain aspect ratio for the logo */
    border-radius: 10px; /* Added slight rounding for a professional touch */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow for emphasis */
}

    .header-container h1 {
    font-size: 32px; /* Increased font size for better visibility */
    margin: 0;
    padding: 0;
    color: #222; /* Slightly darker text for contrast */
    font-family: 'Arial', sans-serif; /* Updated font for a modern look */
    line-height: 1.2; /* Adjusted line height for readability */
}

"""
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
 

# HTML templates for chatbot UI
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" style="max-height: 80px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''





# Load environment variables from Streamlit secrets
LANGCHAIN_ENDPOINT = st.secrets["env"]["LANGCHAIN_ENDPOINT"]
LANGCHAIN_API_KEY = st.secrets["env"]["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = st.secrets["env"]["LANGCHAIN_PROJECT"]
MISTRAL_API_KEY = st.secrets["env"]["MISTRAL_API_KEY"]
PINECONE_API_KEY = st.secrets["env"]["PINECONE_API_KEY"]

# Initialize the LLM (Language Model) with the system prompt in English
system_prompt = """ Dobrodošli u Paragraf Lex! Ovde sam da vas vodim kroz sva pitanja koja imate o PDV-u i elektronskom fakturisanju u Srbiji. Čime vam mogu pomoći danas?

Opis uloge:

Ja sam virtuelni asistent iz Paragraf Lex-a specijalizovan za elektronsko fakturisanje i zakonodavstvo o Porezu na Dodatu Vrednost (PDV) u Republici Srbiji, koristeći informacije iz Paragraf online pravne biblioteke. Moj cilj je da korisnicima pružim jasne, detaljne i tačne informacije koje prevazilaze prethodne primere kvaliteta.

Uputstva za odgovor:

Integracija članaka: Koristiću relevantne delove dostavljenih članaka (segmente) vezane za pitanje korisnika. Citiraću ili referencirati specifične delove zakona, članaka ili klauzula iz ovih članaka kada je to potrebno.

Struktura odgovora:

Kratki uvod: Potvrdiću svoje razumevanje pitanja.

Detaljan odgovor: Pružiću sveobuhvatne i lako razumljive informacije, referencirajući dostavljene članke i regulative.

Pravne reference: Citiraću specifične zakone, članke i klauzule kada je to relevantno.

Zaključak: Ponudiću dodatnu pomoć ili pojašnjenje ako je potrebno.

Prevencija grešaka:

Proveriću tačnost informacija pre nego što ih pružim.

Izbegavaću pretpostavke; ako nedostaju informacije, ljubazno ću tražiti pojašnjenje.

Neću pružati netačne ili zastarele informacije.

Opseg odgovora:

Dozvoljene teme: Elektronsko fakturisanje, PDV, relevantni srpski zakoni i regulative.

Nedozvoljene teme: Pitanja koja nisu vezana za elektronsko fakturisanje ili PDV u Srbiji. Za takva pitanja ljubazno ću objasniti ovo ograničenje.

Stil komunikacije:

Biću profesionalan, prijateljski i pristupačan.

Koristiću jednostavan jezik dostupan korisnicima bez pravnog ili računovodstvenog znanja.

Jasno ću objasniti tehničke termine.

Doslednost jezika: Odgovaraću na istom jeziku na kojem je postavljeno pitanje.

Integracija članaka (segmenti):

Kada korisnik postavi pitanje, sistem će pružiti relevantne članke iz Paragraf online pravne biblioteke kao kontekstualne podatke (segmente) koje ću koristiti za formulisanje odgovora.

Napomene:

Kombinovaću informacije iz dostavljenih podataka (segmenti), svog znanja i relevantnih zakona za najtačniji odgovor.

Uvek ću uzimati u obzir najnovije izmene i ažuriranja zakona i regulativa.

Predstaviću informacije kao potpune odgovore bez spominjanja korišćenja segmenata ili internih izvora.

Cilj:


Moj cilj je da korisnicima pružim najkvalitetnije i najdetaljnije informacije kako bi razumeli i ispunili svoje pravne obaveze vezane za elektronsko fakturisanje i PDV u Republici Srbiji.."""

 

llm = ChatMistralAI(model="mistral-large-latest", system_message=system_prompt)

# Initialize Pinecone for vector database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"  # Ensure this matches your Pinecone environment
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Connect to Pinecone index
index_name = "electronicinvoice1"
index = pc.Index(index_name)

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
)

# Create Pinecone vectorstore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',
    namespace="text_chunks"
)

# Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Define the query refinement prompt template in English
refinement_template = """Create a focused Serbian search query for the RAG retriever bot. Convert to Serbian language if not already. Include key terms, synonyms, and domain-specific vocabulary. Remove filler words. Output only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(
    input_variables=["original_question"],
    template=refinement_template
)

# Create an LLMChain for query refinement using RunnableLambda
refinement_chain = refinement_prompt | llm

# Combine the system prompt with the retrieval prompt template in English
combined_template = f"""{system_prompt}

Please answer the following question using only the context provided:
{{context}}

Question: {{question}}
Answer:"""

retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

# Create a retrieval chain with the combined prompt
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": retrieval_prompt}
)

def process_query(query: str):
    try:
        # Refine the query
        refined_query_msg = refinement_chain.invoke({"original_question": query})
        
        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # Use the refined query in the retrieval chain
        response_msg = retrieval_chain.invoke(refined_query)

        # Corrected extraction of the response
        if isinstance(response_msg, dict):
            response = response_msg.get("result", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit app
# Streamlit app
def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col6:
    pg_logo = add_logo(logo_path="logo.jpg", width=60, height=40)
    st.image(pg_logo)







st.markdown("""
    <h1 style="text-shadow: 2px 2px 5px red; font-weight: bold; text-align: center;">
         💲Paragraf Lex Chatbot📝
    </h1>
""", unsafe_allow_html=True)

st.markdown("""
    <p style="font-size: 18px; color: #000000; line-height: 1.6; text-align: center;">
        👋 Welcome to <strong>Paragraf Lex</strong>, your trusted guide for navigating 🇷🇸 VAT and 🧾 electronic invoicing in Serbia. <br>
        💡 Let me assist you with any questions or provide insights on 📜 legal regulations and more.
    </p>
""", unsafe_allow_html=True)

# Sidebar with suggestion prompts
st.sidebar.title("Common Query")

prompts = [
    "1. Kada se faktura automatski odobrava, a kada automatski odbija?",
    "2. Da li se elektronske fakture šalju u PDF formatu, email-om?",
    "3. Šta je Sistem Elektronskih Faktura (SEF)?",
    "4. Šta je XML format i kako se pravi?",
]

for prompt in prompts:
    st.sidebar.write(prompt)
 

# Session state to save chat history if needed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form
query = st.text_input("Vaše pitanje:")
if st.button("Pošalji") and query:
    response = process_query(query)
    st.session_state.chat_history.append({"question": query, "answer": response})

# Display chat history
for entry in st.session_state.chat_history:
    st.markdown(user_template.replace("{{MSG}}", entry["question"]), unsafe_allow_html=True)
    st.markdown(bot_template.replace("{{MSG}}", entry["answer"]), unsafe_allow_html=True)
    st.write("---")

# Option to clear chat history
if st.button("Obriši istoriju razgovora"):
    st.session_state.chat_history = [] 
