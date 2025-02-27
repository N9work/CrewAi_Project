from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class Trip():
    """Trip crew"""

    # สร้าง Agent ตรงๆ ใน Python (แทน YAML)
    @agent
    def researcher(self) -> Agent:
        return Agent(
            role="Researcher",
            goal="Gather information on the latest industry trends",
            backstory="An expert researcher with years of experience in data analysis.",
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            role="Reporting Analyst",
            goal="Compile and present findings in a structured report",
            backstory="A skilled analyst specializing in data interpretation.",
            verbose=True
        )

    # สร้าง Task ตรงๆ ใน Python (แทน YAML)
    @task
    def research_task(self) -> Task:
        return Task(
            agent=self.researcher(),
            description="Conduct research on the latest trends in AI.",
            expected_output="A detailed report on AI advancements."
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            agent=self.reporting_analyst(),
            description="Compile research findings into a structured markdown report.",
            expected_output="A structured markdown file with research results.",
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Trip crew"""
        return Crew(
            agents=self.agents,  # ดึง Agents จาก @agent decorator
            tasks=self.tasks,  # ดึง Tasks จาก @task decorator
            process=Process.sequential,
            verbose=True
        )
