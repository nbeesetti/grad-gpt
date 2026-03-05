from urllib import response

import gradio as gr
from dotenv import load_dotenv
import plotly.express as px
from supabase import create_client, Client
from coordinator import process_message
import os
import requests
import pandas as pd

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

API_BASE = "http://localhost:8000"

def fetch_notifications(user_id):
    response = requests.get(f"{API_BASE}/notifications/{user_id}")
    data = response.json()

    formatted = []

    for n in data:
        rule = n["NotificationRules"]

        # Use month/day if available, else fallback to text due_date
        if rule.get("month") is not None and rule.get("day") is not None:
            due_display = f"{rule['month']:02d}-{rule['day']:02d}"
        elif rule.get("due_date"):
            due_display = rule["due_date"]
        else:
            due_display = ""

        formatted.append([
            n["id"],            # Notification ID
            due_display,        # Computed due date
            rule.get("message"),# Notification message
            n.get("read")       # Read status
        ])

    df = pd.DataFrame(
        formatted,
        columns=["ID", "Due Date", "Message", "Read"]
    )

    return df



def sync_and_refresh(df, email):

    if df is None or email is None:
        return df

    # Mark notifications as read
    for _, row in df.iterrows():
        if row["Read"] is True:
            requests.post(
                f"{API_BASE}/notifications/read/{row['ID']}"
            )

    # Get user_id from email
    response = supabase.table("Users").select("id").eq("email", email).execute()

    if not response.data:
        print("User not found")
        return df

    user_id = response.data[0]["id"]

    # Fetch updated notifications
    return fetch_notifications(user_id)

def save_courses(email, completed, current, planned):
    completed = completed or []
    current = current or []
    planned = planned or []

    # Remove from planned if already in completed or current
    planned = [c for c in planned if c not in completed + current]

    # Fetch user ID
    response = supabase.table("Users").select("id").eq("email", email).execute()
    if not response.data or len(response.data) == 0:
        print("User not found")
        # Return progress chart + current values to avoid blanking
        return update_progress(completed), completed, current, planned

    user_id = response.data[0]["id"]

    # Update DB
    supabase.table("Users").update({
        "completedCourses": completed,
        "currentCourses": current,
        "plannedCourses": planned
    }).eq("id", user_id).execute()

    # Return updated progress chart + updated lists for dropdowns
    return update_progress(completed), completed, current, planned

# Fetch user profile, courses, notifications, and degree progress plot to pre-populate the dashboard
def load_user_state(email):
    
    # Get user profile
    response = supabase.table("Users").select("*").eq("email", email).execute()
    if len(response.data) == 0:
        return [], [], [], pd.DataFrame(columns=["ID", "Due Date", "Message", "Read"])
    
    user = response.data[0]
    user_id = user["id"]
    
    # Load courses
    completed = user.get("completedCourses") or []
    current = user.get("currentCourses") or []
    planned = user.get("plannedCourses") or []
    
    # Fetch notifications from backend 
    notif_df = fetch_notifications(user_id)

    # Get degree progress plot
    progress_plot = update_progress(completed)

    return completed, current, planned, notif_df, progress_plot, user.get("status") or "Undergraduate"

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

    completed, current, planned, notif_df, progress_plot, status_input = load_user_state(email)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        email,
        email,
        start_term,
        graduation_quarter,
        completed, 
        current, 
        planned,
        notif_df,
        progress_plot,
        status_input
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

def update_profile(email, start_term_input, grad_term_input, status_input):
    if not email:
        return "No user logged in."

    supabase.table("Users").update({
        "startTerm": start_term_input,
        "graduationTarget": grad_term_input,
        "status": status_input 
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
                status_input = gr.Dropdown(
                    label="Status",
                    choices=["Undergraduate", "Graduate"],
                    value=""  
                )

                greet_btn = gr.Button("Save Profile")
                greet_btn.click(
                    update_profile,
                    inputs=[user_state, start_term_input, grad_term_input, status_input],
                    outputs=None
                )

            # Notifications
            with gr.Column(scale=2):
                gr.Markdown("## Notifications")

                notif_df = gr.Dataframe(
                    headers=["ID", "Due Date", "Message", "Read"],
                    datatype=["number", "str", "str", "bool"],
                    interactive=True
                )

                update_notif_btn = gr.Button("Update Notifications")

                update_notif_btn.click(
                    sync_and_refresh,
                    inputs=[notif_df, user_state],
                    outputs=notif_df
                )

        # Courses
        gr.Markdown("## Courses")

        with gr.Row():
            with gr.Column(scale=2):
                completed = gr.Dropdown(
                    choices=COURSE_LIST,
                    label="Completed Courses",
                    multiselect=True,
                    value=[]
                )

                current = gr.Dropdown(
                    choices=COURSE_LIST,
                    label="Current Courses",
                    multiselect=True,
                    value=[]
                )

                planned = gr.Dropdown(
                    choices=COURSE_LIST,
                    label="Planned Courses",
                    multiselect=True,
                    value=[]
                )

                update_progress_btn = gr.Button("Update Progress")

            # Degree Progress Chart
            with gr.Column(scale=1):
                gr.Markdown("## Degree Progress")
                progress_plot = gr.Plot()

        update_progress_btn.click(
            save_courses,
            inputs=[user_state, completed, current, planned],
            outputs=[progress_plot, completed, current, planned]
        )

        # Chat
        gr.Markdown("## Ask GradGPT")

        chatbot = gr.Chatbot(label="GradGPT", height=400)
        chat_state = gr.State([])

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask a question...",
                show_label=False,
                scale=8
            )
            send_btn = gr.Button("Send", scale=1)

        # Calls coordinator logic to process message

        def chat_handler(message, history):

            if not message:
                return history, history, ""

            updated_history = process_message(message, history)

            return updated_history, updated_history, ""

        send_btn.click(
            chat_handler,
            inputs=[chat_input, chat_state],
            outputs=[chatbot, chat_state, chat_input]
        )

        chat_input.submit(
            chat_handler,
            inputs=[chat_input, chat_state],
            outputs=[chatbot, chat_state, chat_input]
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
            grad_term_input,
            completed, 
            current,
            planned,
            notif_df,
            progress_plot,
            status_input
        ]
    )


if __name__ == "__main__":
    demo.launch()
