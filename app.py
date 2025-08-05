import os
import fitz
import gradio as gr
from crewai import Agent, Task, Crew, LLM
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM
llm = LLM(
    model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.7
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define Education Agents
content_creator = Agent(
    role="Educational Content Specialist",
    goal="Create engaging lesson plans and learning materials",
    backstory=(
        "Expert instructional designer with 15+ years experience creating effective "
        "educational content for K-12 and higher education. Specializes in making "
        "complex topics accessible and engaging."
    ),
    verbose=True,
    llm=llm,
    memory=memory
)

curriculum_designer = Agent(
    role="Curriculum Architect",
    goal="Design comprehensive learning pathways",
    backstory=(
        "Education specialist with expertise in curriculum development aligned with "
        "international standards (IB, Common Core, Cambridge). Creates scaffolded "
        "learning sequences with multimodal approaches."
    ),
    llm=llm
)

resource_generator = Agent(
    role="Learning Resource Creator",
    goal="Generate interactive learning activities",
    backstory=(
        "Digital learning expert who creates engaging exercises, assessments, "
        "and multimedia resources using evidence-based pedagogical approaches."
    ),
    llm=llm
)

resource_analyzer = Agent(
    role="Educational Resource Analyst",
    goal="Analyze and explain teaching materials",
    backstory=(
        "Instructional coach who helps educators understand and effectively "
        "utilize teaching resources with practical implementation strategies."
    ),
    llm=llm,
    memory=memory
)

# Content Creation Functions
def create_lesson_content(topic):
    task = Task(
        description=(
            f"Create comprehensive lesson plan on: '{topic}'. Include:\n"
            "1. Clear learning objectives\n"
            "2. Engaging hook activity\n"
            "3. Step-by-step instructional sequence\n"
            "4. Differentiated activities\n"
            "5. Assessment methods\n"
            "6. Technology integration ideas"
        ),
        expected_output="Well-structured lesson plan in markdown format",
        agent=content_creator
    )
    crew = Crew(agents=[content_creator], tasks=[task])
    return crew.kickoff()

def design_curriculum(subject):
    task = Task(
        description=(
            f"Design complete curriculum for: '{subject}'. Include:\n"
            "1. Scope and sequence\n"
            "2. Unit plans with duration\n"
            "3. Key learning outcomes\n"
            "4. Resource recommendations\n"
            "5. Assessment framework\n"
            "6. Standards alignment"
        ),
        expected_output="Curriculum document with logical progression and benchmarks",
        agent=curriculum_designer
    )
    crew = Crew(agents=[curriculum_designer], tasks=[task])
    return crew.kickoff()

def generate_learning_resources(topic):
    task = Task(
        description=(
            f"Create interactive resources for: '{topic}'. Include:\n"
            "1. 5 engaging discussion questions\n"
            "2. 3 collaborative activities\n"
            "3. Formative assessment ideas\n"
            "4. Digital tool suggestions\n"
            "5. Multimedia resource links"
        ),
        expected_output="Diverse, ready-to-use classroom resources",
        agent=resource_generator
    )
    crew = Crew(agents=[resource_generator], tasks=[task])
    return crew.kickoff()

def analyze_educational_resource(file):
    if file is None:
        return "Please upload a PDF file first"
    
    doc = fitz.open(file.name)
    full_text = "".join([page.get_text() for page in doc])
    
    task = Task(
        description=(
            f"Analyze this educational resource:\n{full_text}\n\n"
            "Provide:\n1. Key concepts\n2. Grade level\n3. Pedagogical approach\n"
            "4. Teaching strategies\n5. Modification ideas\n6. Complementary resources"
        ),
        expected_output="Comprehensive resource analysis report",
        agent=resource_analyzer
    )
    crew = Crew(agents=[resource_analyzer], tasks=[task])
    return crew.kickoff()

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎓 AI-Powered Educational Content Studio")
    
    with gr.Tab("📝 Lesson Planner"):
        gr.Markdown("## Create Custom Lesson Plans")
        topic_input = gr.Textbox(label="Topic/Subject", placeholder="Enter a topic like 'Fractions' or 'World War II'")
        lesson_output = gr.Textbox(label="Generated Lesson Plan", lines=15)
        lesson_btn = gr.Button("Generate Lesson Plan")
        lesson_btn.click(create_lesson_content, inputs=topic_input, outputs=lesson_output)
    
    with gr.Tab("📚 Curriculum Designer"):
        gr.Markdown("## Design Complete Curriculums")
        subject_input = gr.Textbox(label="Subject Area", placeholder="e.g., Mathematics Grade 5, Biology High School")
        curriculum_output = gr.Textbox(label="Curriculum Framework", lines=15)
        curriculum_btn = gr.Button("Design Curriculum")
        curriculum_btn.click(design_curriculum, inputs=subject_input, outputs=curriculum_output)
    
    with gr.Tab("💡 Activity Generator"):
        gr.Markdown("## Create Teaching Resources")
        resource_topic = gr.Textbox(label="Learning Topic", placeholder="Enter topic for activities and resources")
        resource_output = gr.Textbox(label="Teaching Resources", lines=15)
        resource_btn = gr.Button("Generate Resources")
        resource_btn.click(generate_learning_resources, inputs=resource_topic, outputs=resource_output)
    
    with gr.Tab("🔍 Resource Analyzer"):
        gr.Markdown("## Analyze Educational Materials")
        file_input = gr.File(label="Upload Teaching Material (PDF)", file_types=[".pdf"])
        analysis_output = gr.Textbox(label="Analysis Report", lines=15)
        analysis_btn = gr.Button("Analyze Resource")
        analysis_btn.click(analyze_educational_resource, inputs=file_input, outputs=analysis_output)

# Launch the app
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)