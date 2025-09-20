import asyncio
from dotenv import load_dotenv
from browser_use import Agent, BrowserSession
from playwright.async_api import async_playwright
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_KEY") # Define this in your .env

openai_api_key =  os.getenv("OPENAI_API_KEY")

#sensitive_data = {'email': os.getenv("EMAIL") , 'password': os.getenv("PASSWORD")}

'''sensitive_data = {
    'https://dev.to': {
        'email': os.getenv("EMAIL"),
        'password': os.getenv("PASSWORD")
    },
}'''

async def main():
    async with async_playwright() as playwright:
        #browser_session = BrowserSession(user_data_dir= None)

        llm = ChatOpenAI(model="gpt-4o", temperature=0.2,openai_api_key=openai_api_key)
        #lm_google = Chat(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
        #browser_session = BrowserSession(allowed_domains=['https://dev.to'], user_data_dir= None)
        browser_session = BrowserSession(user_data_dir= None)

        agent = Agent(
            #task='''go to https://dev.to/ and login with Email and Password then search for AI Agents in the search bar and press enter and then print the title of first 5 blogs''',
            task = """find the top 3 scientists who worked on reinforcement learning in the last 2 years, check their LinkedIn profiles and read their latest abstract  of research paper from a website called arXiv, and explain it to me in simple steps including the maths in a way I can understand
            """,
            llm=llm,
            browser_session=browser_session,
            #sensitive_data=sensitive_data,
            use_vision=False,
        )

        await agent.run()

# To run the main function:
asyncio.run(main())