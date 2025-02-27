from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool
import markdown
import os

# Set your API Key for SerperDevTool (Search Tool)
search_tools = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

app = FastAPI()

# Allow Frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model for receiving the request ----------
class TaskRequest(BaseModel):
    task_type: str   # The selected province
    style: str       # Travel style
    cost: str        # Budget range (Low, Mid, High, Any)
    day: str         # Number of days for the trip
    adults: str      # Number of people
    Rq: str          # Additional requirements from the user (if any)

# ---------- Task mapping for each province ----------
task_mapping = {
    "Phatthalung": {
        "description": (
            "You are responsible for writing the travel package report for Phatthalung. "
            "You must review the information obtained from the Researcher and expand it in detail, "
            "creating 5 travel packages with daily activity timelines, calculating the budget, "
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Phatthalung with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB, and specify recommended accommodation names."
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
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Trang with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB, and specify recommended accommodation names."
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
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Satun with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB, and specify recommended accommodation names."
        ),
        "goal": (
            "Research up-to-date (2024-2025) travel information in "
            "Satun, Thailand."
            "Include another place not only Koh Lipe"
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
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Songkhla with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB, and specify recommended accommodation names."
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
            "and suggesting accommodation options."
        ),
        "expected_output": (
            "Write 5 travel packages for Yala with day-by-day schedules. "
            "Provide all details according to the Researcher's findings, "
            "show approximate costs in THB, and specify recommended accommodation names."
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
            "Suitable for those who enjoy relaxing in cafes like Cafe Name, looking for beautiful photo spots, and sampling light meals."
            "Just replace Cafe Name with the actual name of the café you found . For example:"
            "Suitable for those who enjoy relaxing in cafes like Café Vortex, looking for beautiful photo spots, and sampling light meals."
        )
    },
    "Historical": {
        "description": (
            "Suitable for those interested in history, culture, "
            "and traditional attractions."
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
            "A budget-friendly trip focusing on cheaper travel options and "
            "accommodation to save costs."
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

@app.post("/set_task")
async def set_task(request: TaskRequest):
    """
    Receive the Task from the Frontend → create a CrewAI workflow → return the result.
    There are 2 Agents: Researcher and Writer.
    """

    # Check the province (task_type) provided by the user
    task_info = task_mapping.get(request.task_type)
    if not task_info:
        raise HTTPException(status_code=400, detail="Invalid task type")

    # Check style, cost, day
    style_info = style_mapping.get(request.style)
    if not style_info:
        raise HTTPException(status_code=400, detail="Invalid style option")

    cost_info = cost_mapping.get(request.cost)
    if not cost_info:
        raise HTTPException(status_code=400, detail="Invalid cost option")

    day_info = day_mapping.get(request.day)
    if not day_info:
        raise HTTPException(status_code=400, detail="Invalid day option")

    # ---------- Create the Researcher Agent ----------
    researcher = Agent(
        role="Thai Tour Researcher",
        goal=(
            task_info["goal"]
            + task_info["person"].format(adults=request.adults)
            + task_info["rq"].format(Rq=request.Rq)
        ),
        backstory=(
            "You are an expert in researching travel information in Southern Thailand, "
            "familiar with travel routes in every province in the region. "
            "You can find the latest details (2024-2025) thoroughly."
        ),
        tools=[search_tools],  # Allow the researcher to use the search tool
        verbose=True
    )

    # ---------- Create the Trip Planner (Writer) Agent ----------
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
    search_query = (
        f"Find tourist attractions, activities, accommodations, and foods for a {request.style} trip in "
        f"{task_info['pv']} with a budget of {request.cost} for {request.day} days, "
        f"considering the number of travelers ({request.adults} adults)."
     )

    # ---------- Task 1: Researcher gathers information ----------
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
            "Information on tourist attractions, historical background, "
            "activities, prices, accommodation (including names), food, and transportation, "
            "aligned with the budget: " + request.cost + " and duration: " + request.day + " day(s)."
        ),
        tools=[search_tools],
        # context=[search_query]
    )

    # ---------- Task 2: Writer creates travel packages ----------
    writer_task = Task(
        agent=writer,
        description=task_info["description"],        # Use description from task_mapping
        expected_output=task_info["expected_output"],# Use expected_output from task_mapping
        context=[research_task]                      # Pull the Researcher's results
    )

    # ---------- Create the Crew and run sequentially ----------
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writer_task],
        process=Process.sequential
    )

    # Start the workflow
    result = crew.kickoff()
    html_result = markdown.markdown(str(result))

    return {
        "message": "Task completed!",
        "result": str(html_result)
    }
