import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from io import BytesIO
import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PIL import Image
# Function to retrieve the Google API key from AWS Secrets Manager
def get_secret():
    secret_name = "GOOGLE_API_KEY"
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)

    try:
        # Retrieve the secret value
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        return secret
    except ClientError as e:
        # Handle specific errors such as secret not found or permission issues
        print(f"Failed to retrieve secret: {e}")
        return None

# Set the Google API key as an environment variable
def set_secret_as_env_var():
    secret = get_secret()
    if secret:
        os.environ['GOOGLE_API_KEY'] = secret
    else:
        print("Google API key not found in Secrets Manager.")

# Set the secret as an environment variable
set_secret_as_env_var()

# Load environment variables from .env file


# load_dotenv()
# Fetch the API key from environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")


# Configure the API client with the Google API key
if google_api_key:
    genai.configure(api_key=google_api_key)
    print("Google API key successfully loaded and configured.")
else:
    print("Failed to load Google API key.")

# MODEL = genai.GenerativeModel('gemini-1.5-flash')
text_data = []
def get_pdf_text(pdfs):
    
    for pdf in pdfs:
        try:
            pdf_bytes = BytesIO(pdf.read())
            pdf_reader = PdfReader(pdf_bytes)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text_data.append({
                        'pdf_name': pdf.name,
                        'page_num': page_num + 1,
                        'text': text
                    })
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            

    return text_data


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap = 1000)
    chunks = []
    for data in text_data:
        split_texts = text_splitter.split_text(data['text'])
        for text in split_texts:
            chunks.append({
                'pdf_name': data['pdf_name'],
                'page_num': data['page_num'],
                'text': text
            })
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(
        [chunk['text'] for chunk in text_chunks],
          embedding=embeddings,
           metadatas=[{'pdf_name': chunk['pdf_name'], 'page_num': chunk['page_num']} for chunk in text_chunks]
          )
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the Question as detailed as possible from the provided context from uploaded file. Don't provide wrong answer.
    Context: \n{context}?\n
    Question: \n{question}\n
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['conext', 'questions'])
    chain = load_qa_chain(model, chain_type="stuff", prompt= prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
          return_only_outputs=True)
    
    # Display the response with PDF name and page number
    
    # Consolidate PDF names and page numbers
    pdf_pages = {}
    for doc in docs:
        pdf_name = doc.metadata.get('pdf_name')
        page_num = doc.metadata.get('page_num')
        if pdf_name in pdf_pages:
            pdf_pages[pdf_name].append(page_num)
        else:
            pdf_pages[pdf_name] = [page_num]
    
    # Display the consolidated results
    for pdf_name, pages in pdf_pages.items():
        page_list = ', '.join(map(str, sorted(set(pages))))
        st.write(f"Found in: {pdf_name}, Pages: {page_list}")
    
    # Display the response text
    st.write("Reply:", response["output_text"]) # Print the output text from the response list
 

#######################
## INVOICE READ
def get_response(input, image, prompt):
    response = genai.GenerativeModel('gemini-1.5-flash').generate_content([input, image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # read file into the bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No File Uploaded!!!")



# Streamlit setup
def main():
    st.set_page_config(page_title="Chat with Multiple PDF", layout="wide")
    

   
    
   
     # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["AI Mental Assistant Chatbot","PDF Search", "Invoice Read"])
    
    if page == "PDF Search":
        st.title("Chat with Multiple PDFs using Gemini")
        user_question = st.text_input("Ask a Question from the PDF Files")
    
        if st.button("Search") or user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("Menu:")
            pdf_docs = st.file_uploader("Upload your PDF Files", type="pdf", accept_multiple_files=True)
            if st.button("Submit & Process"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")
                else:
                    st.warning("Please upload at least one PDF file.")
    elif page == "Invoice Read":
        st.title("Gemini Inovice Reader")
        input = st.text_input("Input Prompt:", key = "input")

        uploaded_file = st.file_uploader("Choose a image..", type= ["jpeg", "jpg", "png"])
        image = ""
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image:", use_column_width = True)

        submit = st.button("Tell me about the invoice!!!")

        input_prompts = """
        You are an expert in understanding invoices. We will upload
        a image as invoice and you will have to answer any questions based ont eh uploaded invoice image.
        """


        # if submit button is clicked

        if submit:
            image_data = input_image_setup(uploaded_file = uploaded_file)
            response = get_response(input_prompts, image_data, input)
            st.subheader('the Response is:')
            st.write(response)

    elif page == "AI Mental Assistant Chatbot":
        
        def get_gemini_response(question):
            """
            Gets a response from the Gemini model based on user message and chat history.

            Args:
            user_message (str): The user's message input.
            chat_history (str, optional): The chat history for context. Defaults to "".

            Returns:
            str: The response from the Gemini model.
            """

            # Prompt Template
            prompt_template = """
            You are a friendly, compassionate mental wellness assistant trained to provide supportive and helpful responses. 
            You are not a licensed therapist, so you should avoid giving medical or diagnostic advice. Instead, offer 
            kindness, encouragement, and general well-being tips. 

            Always use positive and calming language. Avoid any response that could cause harm or encourage dangerous behavior. 
            If a user mentions serious distress, gently suggest they seek professional support.

            This is the chat history:
            {chat_history}

            User Message: {user_message}

            Respond in a calm, empathetic, and supportive way while keeping boundaries clear. 
            """

            model = genai.GenerativeModel("gemini-1.5-flash")
            chat = model.start_chat(history=[])
            response = chat.send_message(question)

            return response



        # st.set_page_config(page_title="AI Mental Assistant")
        # Center the header
        st.title("AI Mental Assistant ChatBot", anchor="middle")


        # Initialize session state for chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Create a container for the chat history   and input
        chat_container = st.container()
        with chat_container:
            for role, text in st.session_state['chat_history']:
                if role == "You":
                    st.markdown(f'<div style="text-align: right;"><b>You:</b> {text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="text-align:left; background-color: #e8f0fe; padding: 10px; border-radius: 5px;"><b>Bot:</b> {text}</div>', unsafe_allow_html=True)

        input_container = st.container()
        with input_container:
            input_text = st.text_input("Input:", key="input")
            submit = st.button("Submit")

        if submit and input_text:

            response = get_gemini_response(input_text)
            ## Add user query and response to session chat history
            st.session_state['chat_history'].append(("You", input_text))
            st.subheader("The Response is")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text)) 



            # Scroll to the bottom using JavaScript
            st.markdown(
            '<script>document.querySelector(".st-container").scrollTo(0, document.querySelector(".st-container").scrollHeight);</script>',
            unsafe_allow_html=True
            )

   

if __name__ == "__main__":
    main()
