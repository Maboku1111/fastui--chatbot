import asyncio
from typing import AsyncIterable, Annotated
from decouple import config
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastui import prebuilt_html, FastUI, AnyComponent
from fastui import components as c
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import PageEvent, GoToEvent
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
# import libraries for vector database
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
# import libraries for database
from cohere import Client
from rich.table import Table
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Initialize Cohere client with your API key
cohere_client = Client(api_key=config('COHERE_API_KEY'))

# Create an instance of the Pinecone class
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create an index with a dimension of 1536 and the "euclidean" metric
index_name = "index0"

# Initialize the Pinecone client
index = pc.Index("index0")

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create a database engine
engine = create_engine('sqlite:///chat_history.db')

# Define a base class for declarative class definitions
Base = declarative_base()

# Define a class to represent the chat history table
class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    message = Column(String)
    response = Column(String)

# Create the table
Base.metadata.create_all(engine)

# Create a session maker
Session = sessionmaker(bind=engine)

# Create a table to display data
table = Table(title="Chat History")

# Add columns to the table
table.add_column("Message", style="cyan", no_wrap=True)
table.add_column("Response", style="magenta", no_wrap=True)

# Function to append messages to the table
def append_to_table(message: str, response: str):
    # Add row to the table
    table.add_row(message, response)
    # Add row to the database
    session = Session()
    chat_history = ChatHistory(message=message, response=response)
    session.add(chat_history)
    session.commit()
    session.close()

# Function to display the table
def display_table():
    console = Console()
    console.print(table)

# Function to retrieve chat history from the database and populate the table
def populate_table():
    session = Session()
    chat_history = session.query(ChatHistory).all()
    for entry in chat_history:
        table.add_row(entry.message, entry.response)
    session.close()

# Create the app object
app = FastAPI()

# Implement websockets
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")

# Message history
app.message_history = []

# Message history model
class MessageHistoryModel(BaseModel):
    message: str = Field(title='Message')

# Chat form
class ChatForm(BaseModel):
    chat: str = Field(title=' ', max_length=1000)

# Root endpoint
@app.get('/api/', response_model=FastUI, response_model_exclude_none=True)
def api_index(chat: str | None = None, reset: bool = False) -> list[AnyComponent]:
    if reset:
        app.message_history = []
    return [
        c.PageTitle(text='FastUI Chatbot'),
        c.Page(
            components=[
                # Header
                c.Heading(text='FastUI Chatbot'),
                c.Paragraph(text='This is a simple chatbot built with FastUI and MistralAI.'),
                # Chat history
                c.Table(
                    data=app.message_history,
                    data_model=MessageHistoryModel,
                    columns=[DisplayLookup(field='message', mode=DisplayMode.markdown, table_width_percent=100)],
                    no_data_message='No messages yet.',
                ),
                # Chat form
                c.ModelForm(model=ChatForm, submit_url=".", method='GOTO'),
                # Reset chat
                c.Link(
                    components=[c.Text(text='Reset Chat')],
                    on_click=GoToEvent(url='/?reset=true'),
                ),
                # Chatbot response
                c.Div(
                    components=[
                        c.ServerLoad(
                            path=f"/sse/{chat}",
                            sse=True,
                            load_trigger=PageEvent(name='load'),
                            components=[],
                        )
                    ],
                    class_name='my-2 p-2 border rounded'),
            ],
        ),
        # Footer
        c.Footer(
            extra_text='Made with FastUI',
            links=[]
        )
    ]

# SSE endpoint
@app.get('/api/sse/{prompt}')
async def sse_ai_response(prompt: str) -> StreamingResponse:
    # Check if prompt is empty
    if prompt is None or prompt == '' or prompt == 'None':
        return StreamingResponse(empty_response(), media_type='text/event-stream')
    return StreamingResponse(ai_response_generator(prompt), media_type='text/event-stream')

# Empty response generator
async def empty_response() -> AsyncIterable[str]:
    # Send the message
    m = FastUI(root=[c.Markdown(text='')])
    msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
    yield msg
    # Avoid the browser reconnecting
    while True:
        yield msg
        await asyncio.sleep(10)



# MistralAI response generator
async def ai_response_generator(prompt: str) -> AsyncIterable[str]:
    # Mistral client
    mistral_client = MistralClient(api_key=config('MISTRAL_API_KEY'))
    system_message = "You are a helpful chatbot. You will help people with answers to their questions."
    # Output variables
    output = f"**User:** {prompt}\n\n"
    msg = ''
    # Vectorize the prompt
    prompt_vector = model.encode([prompt]).tolist()
    # Query the vector database
    results = index.query(vector=prompt_vector, top_k=5, include_metadata=True)
    # Extract the relevant information from the query results
    relevant_info = "\n".join([result["metadata"]["text"] for result in results["matches"]])
    # Incorporate the relevant information into the prompt template
    prompt_template = "Previous messages:\n"
    for message_history in app.message_history:
        prompt_template += message_history.message + "\n"
    prompt_template += f"Relevant information:\n{relevant_info}\nHuman: {prompt}"

    # Mistral chat messages
    mistral_messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(role="user", content=prompt_template)
    ]
    # Stream the chat
    output += f"**Chatbot:** "
    for chunk in mistral_client.chat_stream(model="mistral-small", messages=mistral_messages):
        if token := chunk.choices[0].delta.content or "":
            # Add the token to the output
            output += token
            # Send the message
            m = FastUI(root=[c.Markdown(text=output)])
            msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
            yield msg
    # Append the message to the history
    message = MessageHistoryModel(message=output)
    app.message_history.append(message)
    # Avoid the browser reconnecting
    while True:
        yield msg
        await asyncio.sleep(10)

# Cohere response generator
async def cohere_response_generator(prompt: str) -> AsyncIterable[str]:
    # Use Cohere to generate a response
    response = cohere_client.generate(
        model='large',
        prompt=prompt,
        max_tokens=50,
        temperature=0.5
    )
    output = f"**User:** {prompt}\n\n**Chatbot (Cohere):** {response.generations[0].text}"
    # Send the message
    m = FastUI(root=[c.Markdown(text=output)])
    msg = f'data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n'
    yield msg
    # Append the message to the history
    message = MessageHistoryModel(message=output)
    app.message_history.append(message)
    # Avoid the browser reconnecting
    while True:
        yield msg
        await asyncio.sleep(10)

# SSE endpoint
@app.get('/api/sse/{prompt}')
async def sse_ai_response(prompt: str) -> StreamingResponse:
    # Check if prompt is empty
    if prompt is None or prompt == '' or prompt == 'None':
        return StreamingResponse(empty_response(), media_type='text/event-stream')
    # Use Cohere response generator
    return StreamingResponse(cohere_response_generator(prompt), media_type='text/event-stream')

# Pre-built HTML
@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))
