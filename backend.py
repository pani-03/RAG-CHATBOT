import os
import sqlite3
import requests
import pytz
import wikipedia
from dotenv import load_dotenv
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash"
)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations (add, sub, mul, div)."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_weather(city: str) -> dict:
    """Fetch current weather information for a given city using the OpenWeather API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OpenWeather API key not found"}
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return {"error": f"Could not fetch weather data: {data.get('message', 'Unknown error')}"}
        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "description": data["weather"][0]["description"]
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_current_time(timezone: str = "UTC") -> dict:
    """Get the current time in a given timezone (default UTC)."""
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return {
            "timezone": timezone,
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def wikipedia_search(query: str, sentences: int = 3) -> dict:
    """Search Wikipedia for a query and return a summary."""
    try:
        summary = wikipedia.summary(query, sentences=sentences)
        return {
            "query": query,
            "summary": summary
        }
    except wikipedia.DisambiguationError as e:
        return {"error": f"Disambiguation needed: {e.options[:5]}"}
    except wikipedia.PageError:
        return {"error": f"No Wikipedia page found"}
    except Exception as e:
        return {"error": str(e)}

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount from one currency to another using ExchangeRate API."""
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key:
        return {"error": "ExchangeRate API key not found"}
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency.upper()}/{to_currency.upper()}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["result"] == "success":
            conversion_rate = data["conversion_rate"]
            converted_amount = amount * conversion_rate
            return {
                "conversion_rate": conversion_rate,
                "converted_amount": round(converted_amount, 2)
            }
        else:
            return {"error": f"Currency conversion failed"}
    except Exception as e:
        return {"error": str(e)}

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert values between units (length, weight, temperature)."""
    length_factors = {"meters": 1, "kilometers": 1000, "miles": 1609.34, "feet": 0.3048}
    weight_factors = {"grams": 1, "kilograms": 1000, "pounds": 453.592}
    def c2f(c): return (c * 9/5) + 32
    def f2c(f): return (f - 32) * 5/9
    def c2k(c): return c + 273.15
    def k2c(k): return k - 273.15
    try:
        if from_unit in ["celsius", "fahrenheit", "kelvin"] and to_unit in ["celsius", "fahrenheit", "kelvin"]:
            if from_unit == "celsius": temp_c = value
            elif from_unit == "fahrenheit": temp_c = f2c(value)
            elif from_unit == "kelvin": temp_c = k2c(value)
            if to_unit == "celsius": result = temp_c
            elif to_unit == "fahrenheit": result = c2f(temp_c)
            elif to_unit == "kelvin": result = c2k(temp_c)
            return {"converted_value": round(result, 2)}
        elif from_unit in length_factors and to_unit in length_factors:
            result = value * length_factors[from_unit] / length_factors[to_unit]
            return {"converted_value": round(result, 2)}
        elif from_unit in weight_factors and to_unit in weight_factors:
            result = value * weight_factors[from_unit] / weight_factors[to_unit]
            return {"converted_value": round(result, 2)}
        else:
            return {"error": "Unsupported conversion"}
    except Exception as e:
        return {"error": str(e)}

@tool
def translate_text(text: str, target_lang: str = "hi") -> dict:
    """Translate text into a target language (default Hindi)."""
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl={target_lang}&dt=t&q={text}"
        response = requests.get(url)
        translated_text = response.json()
        return {"original": text, "translated": translated_text, "target_lang": target_lang}
    except Exception as e:
        return {"error": str(e)}

@tool
def pdf_text_extractor(file_path: str) -> dict:
    """Extract text from a PDF file (first 1000 characters)."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return {"text": text[:1000]}
    except Exception as e:
        return {"error": str(e)}

@tool
def youtube_transcript(video_url: str, lang: str = "en") -> dict:
    """Extract transcript from a YouTube video in the specified language."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        video_id = video_url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        transcript_text = " ".join([s['text'] for s in transcript])
        return {"transcript": transcript_text[:1000]}
    except Exception as e:
        return {"error": str(e)}

# List of tools
tools = [
    search_tool, calculator, get_weather, get_current_time, wikipedia_search,
    currency_converter, unit_converter, translate_text, pdf_text_extractor, youtube_transcript
]

llm_with_tools = llm.bind_tools(tools)

class ChatState(dict):
    """State representation for the chatbot graph."""
    messages: list

def chat_node(state: ChatState):
    """Chat node that invokes the LLM with tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    """Retrieve all saved conversation thread IDs from the database."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
