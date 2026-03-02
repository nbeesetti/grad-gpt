from fastapi import FastAPI
from services.notif import (
    generate_notifications,
    mark_as_read
)
from services.user import *
from pydantic import BaseModel
from typing import List

app = FastAPI()

@app.get("/notifications/{user_id}")
def get_notifications(user_id: str):
    return generate_notifications(user_id)

@app.post("/notifications/read/{notification_id}")
def read_notification(notification_id: str):
    mark_as_read(notification_id)
    return {"message": "Notification marked as read"}

class CourseUpdate(BaseModel):
    completed: List[str]
    current: List[str]
    planned: List[str]

@app.post("/users/{user_id}/courses")
def update_user_courses(user_id: int, payload: CourseUpdate):

    handle_term_transition(user_id)

    return update_courses_in_db(user_id, payload)