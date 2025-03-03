from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool
from fastapi.staticfiles import StaticFiles
import markdown
import os
import requests
import json

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:

    print("Warning: SERPER_API_KEY is not set. Please set it in your environment.")
    
# ตั้งค่า SerperDevTool (Search Tool)
search_tools = SerperDevTool(api_key=SERPER_API_KEY)

app = FastAPI()

# Allow Frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Model for receiving the request ----------
class TaskRequest(BaseModel):
    task_type: str 
    style: str      
    cost: str       
    day: str        
    adults: str     
    Rq: str        

# ---------- Task mapping for each province ----------
task_mapping = {
    "All": {
        "description": (
            "You are responsible for writing the travel package report for Phatthalung,Trang,Satun,Songkhla and Yala. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "recommend a signature dish or local food"
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Phatthalung,Trang,Satun,Songkhla and Yala with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB,recommend a local food or signature dish and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Phatthalung,Trang,Satun,Songkhla and Yala, Thailand."
        ),
        "person": "Number of travelers: {adults}\n",
        "rq": "Additional requirements: {Rq}\n",
        "pv": "Destination: Phatthalung,Trang,Satun,Songkhla and Yala, Thailand\n"
    },
    "Phatthalung": {
        "description": (
            "You are responsible for writing the travel package report for Phatthalung. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "recommend a signature dish or local food"
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Phatthalung with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB,recommend a local food or signature dish and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Phatthalung, Thailand."
        ),
        "person": "Number of travelers: {adults}\n",
        "rq": "Additional requirements: {Rq}\n",
        "pv": "Destination: Phatthalung, Thailand\n"
    },
    "Trang": {
        "description": (
            "You are responsible for writing the travel package report for Trang. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "recommend a signature dish or local food"
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Trang with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB,recommend a local food or signature dish and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Trang, Thailand."
        ),
        "person": "Number of travelers: {adults}\n",
        "rq": "Additional requirements: {Rq}\n",
        "pv": "Destination: Trang, Thailand\n"
    },
    "Satun": {
        "description": (
            "You are responsible for writing the travel package report for Satun. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "recommend a signature dish or local food"
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Satun with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB,recommend a local food or signature dish and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Satun, Thailand. Include another place not only Koh Lipe."
        ),
        "person": "Number of travelers: {adults}\n",
        "rq": "Additional requirements: {Rq}\n",
        "pv": "Destination: Satun, Thailand\n"
    },
    "Songkhla": {
        "description": (
            "You are responsible for writing the travel package report for Songkhla. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "recommend a signature dish or local food"
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Songkhla with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB,recommend a local food or signature dish and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Songkhla, Thailand."
        ),
        "person": "Number of travelers: {adults}\n",
        "rq": "Additional requirements: {Rq}\n",
        "pv": "Destination: Songkhla, Thailand\n"
    },
    "Yala": {
        "description": (
            "You are responsible for writing the travel package report for Yala. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "recommend a signature dish or local food"
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Yala with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB,recommend a local food or signature dish and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Yala, Thailand."
        ),
        "person": "Number of travelers: {adults}\n",
        "rq": "Additional requirements: {Rq}\n",
        "pv": "Destination: Yala, Thailand\n"
    },
}

# ---------- Data for Style, Cost, and Day mappings ----------
style_mapping = {
    "Natural": {
        "description": (
            "Suitable for those who want to experience nature, "
            "such as mountains, waterfalls, sea, and various outdoor activities."
        )
    },
    "Cafe": {
        "description": (
            "Suitable for those who enjoy relaxing in cafes like Cafe Name, looking for beautiful photo spots, "
            "and sampling light meals."
        )
    },
    "Historical": {
        "description": (
            "Suitable for those interested in history, culture, and traditional attractions."
        )
    },
    "Extreme Activity": {
        "description": (
            "Suitable for adrenaline seekers who love excitement and challenges, "
            "such as rock climbing, diving, rafting, etc."
        )
    },
    "Any": {
        "description": (
            "No specific travel style specified; can be adapted to various interests."
        )
    }
}

cost_mapping = {
    "Low": {
        "description": (
            "A budget-friendly trip focusing on cheaper travel options and accommodations to save costs."
        )
    },
    "Mid": {
        "description": (
            "A moderate budget that allows you to choose mid-range to good quality "
            "activities and accommodations at reasonable prices."
        )
    },
    "High": {
        "description": (
            "A high budget focusing on a comfortable trip with luxurious accommodation "
            "and premium activities."
        )
    },
    "Any": {
        "description": (
            "No budget limitations; free to select any option as desired."
        )
    }
}

day_mapping = {
    "Any": {
        "description": "No specific number of days; flexible to adjust as needed."
    },
    "1": {
        "description": "A short 1-day trip, traveling in the morning and returning in the evening."
    },
    "2": {
        "description": "A 2-day, 1-night trip, suitable for a weekend getaway."
    },
    "3": {
        "description": "A 3-day, 2-night trip suitable for visiting multiple attractions."
    },
    "4": {
        "description": "A 4-day, 3-night trip for a more immersive travel experience."
    },
    "5": {
        "description": "A 5-day, 4-night trip for those who want to explore in-depth."
    },
    "6": {
        "description": "A 6-day, 5-night trip for well-rounded exploration."
    },
    "7": {
        "description": "A 7-day, 6-night trip to see a variety of places or multiple provinces."
    },
    "more7": {
        "description": (
            "A trip longer than 7 days, suitable for in-depth travel or "
            "extended stay in multiple areas."
        )
    },
}

def search_serper(query: str):
    """
    เรียกใช้งาน Serper (Google Search) และคืนค่าที่เหมาะสม
    """
    if not SERPER_API_KEY:
        raise HTTPException(status_code=400, detail="Serper API key not configured.")

    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        data = response.json()
        
        # ดึงข้อมูลจากผลลัพธ์ organic เท่านั้น
        search_results = data.get("organic", [])

        # แปลงเป็นข้อความที่สามารถใช้เป็น context
        formatted_results = "\n".join([f"- {result['title']}: {result['link']}" for result in search_results])

        return {
            "description": f"ผลการค้นหาสำหรับ: {query}",
            "expected_output": formatted_results
        }
    else:
        raise HTTPException(status_code=response.status_code, detail="Search API request failed.")


@app.post("/set_task")
async def set_task(request: TaskRequest):
    """
    รับ Task จาก Frontend → สร้าง CrewAI workflow → ส่งผลลัพธ์กลับ
    """
    

    # 1) ตรวจสอบ province (task_type)
    task_info = task_mapping.get(request.task_type)
    if not task_info:
        raise HTTPException(status_code=400, detail="Invalid task type")

    # 2) ตรวจสอบ style, cost, day
    style_info = style_mapping.get(request.style)
    if not style_info:
        raise HTTPException(status_code=400, detail="Invalid style option")

    cost_info = cost_mapping.get(request.cost)
    if not cost_info:
        raise HTTPException(status_code=400, detail="Invalid cost option")

    day_info = day_mapping.get(request.day)
    if not day_info:
        raise HTTPException(status_code=400, detail="Invalid day option")

    # 3) สร้าง Researcher Agent
    researcher = Agent(
        role="Thai Tour Researcher",
        goal=(
            task_info["goal"]
            + task_info["person"].format(adults=request.adults)
            + task_info["rq"].format(Rq=request.Rq)
            + style_info["description"]
            + cost_info['description']
            + day_info['description']
        ),
        backstory=(
            "You are an expert in researching travel information in Southern Thailand, "
            "familiar with travel routes in every province in the region. "
            "You can find the latest details (2024-2025) thoroughly."
        ),
        tools=[search_tools],  # researcher สามารถใช้เซิร์จได้
        verbose=True
    )

    # 4) สร้าง Trip Planner (Writer) Agent
    writer = Agent(
        role="Trip Planner",
        goal=(
            "Gather information from the researcher and create 5 travel packages "
            "with schedules and full details for "
            + task_info["pv"]
        ),
        backstory=(
            "You are an expert in planning tours throughout Southern Thailand. "
            "You can create compelling tour packages with comprehensive details "
            "and utilize the researcher's data effectively."
        ),
        verbose=True
    )

    # ค้นหาข้อมูลเบื้องต้นตรงนี้ เพื่อนำไปใส่ใน context
    query = (
        f"Find tourist attractions, activities, accommodations, and foods for a {request.style} trip in "
        f"{task_info['pv']} with a budget of {request.cost} for {request.day} days, "
        f"considering the number of travelers ({request.adults} adults)."
    )
    search_results = search_serper(query)

    # 6) สร้าง Task สำหรับ Researcher
    research_task = Task(
        agent=researcher,
        description=(
            "Conduct research on popular tourist attractions from 2024 to present (2025) in "
            + task_info["pv"]
            + "including activities, recommended foods, accommodations, and costs according to the budget, "
            "while also considering the user's chosen travel style ("
            + request.style
            + ")."
        ),
        expected_output=(
            "Information on tourist attractions, historical background, activities, prices, "
            "accommodation (including names), recommended foods, and transportation, "
            f"aligned with the budget: {request.cost}, day(s): {request.day}."
        ),
        tools=[search_tools],
        context=[search_results],
        # output_file='./output/research/research_output.md'
    )

    # 7) สร้าง Task สำหรับ Writer
    writer_task = Task(
        agent=writer,
        description=task_info["description"],
        expected_output=task_info["expected_output"],
        context=[research_task],  # ดึงผลจาก researcher
        # output_file='./output/writer/writer_output.md'
    )

    # 8) รวม Task ไว้ใน Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writer_task],
        process=Process.sequential,
    )

    # 9) สั่งทำงาน
    result = crew.kickoff()

    html_result = markdown.markdown(str(result))  # แปลง Markdown เป็น HTML

    return {
        "message": "Task completed!",
        "result": str(html_result),
    }
