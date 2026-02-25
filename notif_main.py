from fastapi import FastAPI
from services.notif import (
    generate_notifications,
    mark_as_read
)

app = FastAPI()

@app.get("/notifications/{user_id}")
def get_notifications(user_id: str):
    return generate_notifications(user_id)


@app.post("/notifications/read/{notification_id}")
def read_notification(notification_id: str):
    mark_as_read(notification_id)
    return {"message": "Notification marked as read"}
