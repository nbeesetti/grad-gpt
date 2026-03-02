from db import supabase
from services.notif import get_current_term

def handle_term_transition(user_id):

    current_term = get_current_term()

    user = (
        supabase
        .table("Users")
        .select("completedCourses, currentCourses, lastTermChecked")
        .eq("id", user_id)
        .single()
        .execute()
    ).data

    last_term = user.get("lastTermChecked")

    # If first time, just set term
    if not last_term:
        supabase.table("Users").update({
            "lastTermChecked": current_term
        }).eq("id", user_id).execute()
        return

    # If term has changed
    if last_term != current_term:

        completed = set(user.get("completedCourses") or [])
        current = set(user.get("currentCourses") or [])

        # Move current -> completed
        updated_completed = completed.union(current)

        # Clear current
        updated_current = []

        supabase.table("Users").update({
            "completedCourses": list(updated_completed),
            "currentCourses": updated_current,
            "lastTermChecked": current_term
        }).eq("id", user_id).execute()

def update_courses_in_db(user_id, payload):

    user = (
        supabase
        .table("Users")
        .select("completedCourses, currentCourses, plannedCourses")
        .eq("id", user_id)
        .single()
        .execute()
    ).data

    existing_completed = set(user.get("completedCourses") or [])
    existing_current = set(user.get("currentCourses") or [])
    existing_planned = set(user.get("plannedCourses") or [])

    new_completed = set(payload.completed or [])
    new_current = set(payload.current or [])
    new_planned = set(payload.planned or [])

    # Completed never deletes
    updated_completed = existing_completed.union(new_completed)

    # Current fully replaced
    updated_current = new_current

    # Planned logic
    updated_planned = existing_planned.union(new_planned)
    updated_planned -= updated_completed
    updated_planned -= updated_current

    supabase.table("Users").update({
        "completedCourses": list(updated_completed),
        "currentCourses": list(updated_current),
        "plannedCourses": list(updated_planned)
    }).eq("id", user_id).execute()

    return {"status": "success"}