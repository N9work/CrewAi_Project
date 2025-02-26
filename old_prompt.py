from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Crew, Task, Process 
from crewai_tools import SerperDevTool

search_tools = SerperDevTool()

app = FastAPI()

# ✅ อนุญาตให้ Frontend เรียก API ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    task_type: str
    adults: str
    Rq: str
# ✅ รายละเอียด Task แต่ละประเภท
task_mapping = {
    "Phatthalung": {
        "description": "Review the context you got and expand each topic into a full section for a report.Make sure the report is detailed and contains any and all relevant information.",
        "expected_output": "create a 5 package for Phattalung that include all information,timeline for a tourism and calculate pricing for each things.",
        "goal":"Research tourist attractions in Phatthalung, Thailand and all information is latest in 2024",
        "person":"This trip have {adults} person",
        "rq":"This is an additional requirement for them : {Rq}",
    },
    "Trang": {
        "description": "Review the context you got and expand each topic into a full section for a report.Make sure the report is detailed and contains any and all relevant information.",
        "expected_output": "create a 5 package for Trang that include all information,timeline for a tourism and calculate pricing for each things",
        "goal":"Research tourist attractions in Trang, Thailand and all information is latest in 2024",
        "person":"This trip have {adults} person",
        "rq":"This is an additional requirement for them : {Rq}",
    },
    "Satun": {
        "description": "Review the context you got and expand each topic into a full section for a report.Make sure the report is detailed and contains any and all relevant information.",
        "expected_output": "create a 5 package for Satun that include all information,timeline for a tourism and calculate pricing for each things",
        "goal":"Research tourist attractions in Satun, Thailand and all information is latest in 2024",
        "person":"This trip have {adults} person",
        "rq":"This is an additional requirement for them : {Rq}",
    },
    "Songkhla": {
        "description": "Review the context you got and expand each topic into a full section for a report.Make sure the report is detailed and contains any and all relevant information.",
        "expected_output": "create a 5 package for Songkhla that include all information,timeline for a tourism and calculate pricing for each things",
        "goal":"Research tourist attractions in Songkhla, Thailand and all information is latest in 2024",
        "person":"This trip have {adults} person",
        "rq":"This is an additional requirement for them : {Rq}",
    },
    "Yala": {
        "description": "Review the context you got and expand each topic into a full section for a report.Make sure the report is detailed and contains any and all relevant information.",
        "expected_output": "create a 5 package for Yala that include all information,timeline for a tourism and calculate pricing for each things",
        "goal":"Research tourist attractions in Yala, Thailand and all information is latest in 2024",
        "person":"This trip have {adults} person",
        "rq":"This is an additional requirement for them : {Rq}",
    },

}


class TaskRequest(BaseModel):
    task_type: str
    adults: str
    kids: str = None
    Rq: str
@app.post("/set_task")
async def set_task(request: TaskRequest):
    """รับ Task จาก Frontend → CrewAI kickoff() → ส่งผลลัพธ์กลับ"""

    task_info = task_mapping.get(request.task_type)
    if not task_info:
        raise HTTPException(status_code=400, detail="Invalid task type")

    # สร้าง Agent นักวิจัยข้อมูลท่องเที่ยว
    researcher = Agent(
        role="Thai Tour Researcher",
        goal=task_info["goal"]+task_info["person"].format(adults=request.adults)+"and"+task_info["rq"].format(Rq=request.Rq),
        backstory="You have experience working as a travel guide in  Southen of Thailand.You have go all of Southen in Thailand and you are a good travel reaseacher to describe everything that nedd to know",
        tools=[search_tools],
        verbose=True,
        temperature=0.5
    )

    # สร้าง Agent ผู้เขียนแพ็กเกจทัวร์
    writer = Agent(
        role="Trip Planner",
        goal="Write up a 5 Tour Package for each province and describe all timelines for each package in Trang, Yala, Songkhla, Satun, and Phatthalung, Thailand.",
        backstory="You are an experienced travel package creator,good story teller,well know in Southen of Thailand and you are working in tour company that  most famous in Thailand.",
        verbose=True,
        temperature=0.3
    )

    # **Task 1: ให้ researcher ค้นหาข้อมูลแหล่งท่องเที่ยว**
    research_task = Task(
        agent=researcher,
        description="Conduct comprehensive research about popular destinations"+task_info["goal"],
        expected_output="Gather accurate and detailed information about destinations, potential activities, prices, accommodation options, and transportation details and describe everything for each day."
    )

    # **Task 2: ให้ writer เขียนแพ็กเกจทัวร์โดยใช้ข้อมูลจาก researcher**
    writer_task = Task(
        agent=writer,
        description=task_info["description"],  # ใช้ข้อมูลจาก task_mapping
        expected_output=task_info["expected_output"],
        context=[research_task]  # ใช้ข้อมูลที่ได้จาก research_task
    )

    # สร้าง Crew ให้ทำงานแบบ Sequential
    crew = Crew(
        agents=[researcher, writer], 
        tasks=[research_task, writer_task], 
        process=Process.sequential
    )

    # เริ่มทำงานและรับผลลัพธ์
    result = crew.kickoff()

    return {"message": "Task completed!", "result": str(result)}
