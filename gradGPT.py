import gradio as gr
from dotenv import load_dotenv
import plotly.express as px
from supabase import create_client, Client
import os

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

# Temporary Information
COURSES = {
    "CSC 508": ("Software Engineering I", 4),
    "CSC 509": ("Software Engineering II", 4),
    "CSC 513": ("Computing Education Research", 4),
    "CSC 515": ("Computer Architecture", 4),
    "CSC 521": ("Computer Security", 4),
    "CSC 522": ("Advanced Network Security", 4),
    "CSC 524": ("System Security", 4),
    "CSC 530": ("Languages and Translators", 4),
    "CSC 540": ("Theory of Computation II", 4),
    "CSC 549": ("Advanced Algorithms", 4),
    "CSC 550": ("Operating Systems", 4),
    "CSC 560": ("Database Systems", 4),
    "CSC 564": ("Computer Networks", 4),
    "CSC 566": ("Advanced Data Mining", 4),
    "CSC 569": ("Distributed Computing", 4),
    "CSC 570": ("Topics in CS", 4),
    "CSC 572": ("Computer Graphics", 4),
    "CSC 580": ("Artificial Intelligence", 4),
    "CSC 581": ("Knowledge Management", 4),
    "CSC 582": ("Computational Linguistics", 4),
    "CSC 587": ("Advanced Deep Learning", 4),
    "CSC 590": ("Thesis Seminar", 1),
    "CSC 596": ("Research I", 2),
    "CSC 597": ("Research II", 2),
    "CSC 599": ("Thesis", 4),
}


COURSE_LIST = list(COURSES.keys())
TOTAL_UNITS_REQUIRED = 45

def login_user(email):
    if not email or "@" not in email:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            "",
            "",
            ""
        )

    response = supabase.table("Users").select("*").eq("email", email).execute()

    if len(response.data) == 0:
        # Create user with default graduation quarter
        supabase.table("Users").insert({
            "email": email,
            "graduationTarget": "Spring 2026",
            "startTerm": "Fall 2025"
        }).execute()

        graduation_quarter = "Spring 2026"
        start_term = "Fall 2025"
    else:
        graduation_quarter = response.data[0].get("graduationTarget", "")
        start_term = response.data[0].get("startTerm", "")

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        email,
        email,
        start_term,
        graduation_quarter
    )

# Needs better logic to accurately not just reflect 45 units but matches all the required classes needed
# in the 45 units i.e. need to take 599...
def calculate_units(course_list):
    total = 0

    for course in course_list:
        if course in COURSES:
            total += COURSES[course][1]

    return total

def generate_response(question):
    # Placeholder for LLM / Agent integration
    if question.strip() == "":
        return "Please enter a question."
    
    return f"You asked: '{question}'.\n(This will be handled by GradGPT agents.)"

def update_progress(completed):

    completed_units = calculate_units(completed)
    remaining = max(TOTAL_UNITS_REQUIRED - completed_units, 0)

    fig = px.pie(
        names=["Completed", "Remaining"],
        values=[completed_units, remaining],
        title="Degree Progress",
        color=["Completed", "Remaining"],
        color_discrete_map={
            "Completed": "green",
            "Remaining": "red"
        }
    )

    fig.update_layout(
        height=300,
        width=300,
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=True
    )

    return fig



def style_notifications(df):

    if df is None or len(df) == 0:
        return df

    # Sort unread first
    df = df.sort_values(by="Read", ascending=True)

    return df


def update_notifications(df):

    if df is None:
        return df

    df = style_notifications(df)

    return df

def update_profile(email, start_term_input, grad_term_input):
    if not email:
        return "No user logged in."

    supabase.table("Users").update({
        "startTerm": start_term_input,
        "graduationTarget": grad_term_input
    }).eq("email", email).execute()
    
with gr.Blocks(title="GradGPT Dashboard") as demo:

    user_state = gr.State(None)

    with gr.Column(visible=True) as login_group:
        gr.Markdown("## Sign In")

        email_input = gr.Textbox(label="Email")
        login_btn = gr.Button("Continue")


    with gr.Column(visible=False) as dashboard_group:
        gr.Markdown("# GradGPT Dashboard")


        # Profile
        with gr.Row():

            # User Profile
            with gr.Column(scale=1):

                gr.Markdown("## User Profile")
                email_display = gr.Textbox(label="Email", interactive=False)
                start_term_input = gr.Textbox(label="Start Term",)
                grad_term_input = gr.Textbox(label="Graduation Term")

                greet_btn = gr.Button("Save Profile")
                greet_btn.click(
                    update_profile,
                    inputs=[user_state, start_term_input, grad_term_input],
                    outputs=None
                )


            # Notifications
            with gr.Column(scale=2):
                gr.Markdown("## Notifications")

                notif_df = gr.Dataframe(
                    headers=[
                        "Due Date",
                        "Message", "Read"
                    ],
                    datatype=[
                        "date",
                        "str", "bool"
                    ],
                    value=[
                        ["2026-03-15",
                        "Submit Thesis Proposal", False],

                        ["2026-02-01",
                        "Register for Spring", True]
                    ],
                    interactive=True
                )

                update_notif_btn = gr.Button("Update Notifications")

                update_notif_btn.click(
                    update_notifications,
                    notif_df,
                    notif_df
                )

        # Courses
        gr.Markdown("## Courses")

        with gr.Row():
            with gr.Column(scale=2):
                completed = gr.Dropdown(
                    choices=COURSE_LIST,
                    label="Completed Courses",
                    multiselect=True
                )

                current = gr.Dropdown(
                    choices=COURSE_LIST,
                    label="Current Courses",
                    multiselect=True
                )

                planned = gr.Dropdown(
                    choices=COURSE_LIST,
                    label="Planned Courses",
                    multiselect=True
                )
            
                update_progress_btn = gr.Button("Update Progress")

            # Degree Progress Chart
            with gr.Column(scale=1):
                gr.Markdown("## Degree Progress")
                progress_plot = gr.Plot()



        update_progress_btn.click(
            update_progress,
            completed,
            progress_plot
        )


        # Chat
        gr.Markdown("## Ask GradGPT")

        question = gr.Textbox(
            placeholder="Ask a question..."
        )

        answer = gr.Textbox(
            label="Response",
            lines=4,
            interactive=False
        )

        ask_btn = gr.Button("Submit")

        ask_btn.click(
            generate_response,
            question,
            answer
        )


        # Styling (Notification Colors)
        gr.HTML("""
            <style>

            /* Deadline = Light Red */
            tbody tr td:nth-child(2):contains("Deadline") {
                background-color: #ffe6e6;
            }

            /* Reminder = Light Orange */
            tbody tr td:nth-child(2):contains("Reminder") {
                background-color: #fff0cc;
            }

            </style>
        """)


        gr.Markdown("---")
        gr.Markdown("GradGPT © 2026")

    login_btn.click(
        login_user,
        inputs=email_input,
        outputs=[
            login_group,
            dashboard_group,
            user_state,
            email_display,
            start_term_input,
            grad_term_input
        ]
    )


if __name__ == "__main__":
    demo.launch()
