from datetime import datetime, timedelta, date
from db import supabase

# Fetch user profile
def get_user_profile(user_id):
    response = supabase.table("Users") \
        .select("*") \
        .eq("id", user_id) \
        .single() \
        .execute()
    return response.data

# Fetch all notification rules
def get_notification_rules():
    response = supabase.table("NotificationRules").select("*").execute()
    return response.data

# Check if notification already exists
def notification_exists(user_id, rule_id):
    response = supabase.table("Notifications") \
        .select("id") \
        .eq("userId", user_id) \
        .eq("ruleId", rule_id) \
        .execute()
    return len(response.data) > 0

# Create notification
def create_notification(user_id, rule):
    supabase.table("Notifications").insert({
        "userId": user_id,
        "ruleId": rule["id"],
        "read": False,
        "created_at": datetime.now().isoformat()
    }).execute()

def get_current_term():
    today = date.today()
    year = today.year
    month = today.month

    if month in [9, 10, 11]:
        return f"Fall {year}"
    elif month in [1, 2, 3]:
        return f"Winter {year}"
    elif month in [4, 5, 6]:
        return f"Spring {year}"
    else:
        return f"Summer {year}"
    
def apply_term_offset(term_string, offset):
    term, year = term_string.split()
    year = int(year)

    terms = ["Winter", "Spring", "Summer", "Fall"]
    total_terms = len(terms)  # 4

    index = terms.index(term)
    new_index = index + offset

    # Adjust year when crossing boundaries
    while new_index < 0:
        new_index += total_terms
        year -= 1

    while new_index >= total_terms:
        new_index -= total_terms
        year += 1

    return f"{terms[new_index]} {year}"

# Evaluate all rules for user
def evaluate_rules(user):
    rules = get_notification_rules()
    today = date.today()

    for rule in rules:
        should_trigger = False
        trigger_type = rule.get("trigger_type")

        # if the notification's due date is on an annual date (e.g. every year on oct. 10)
        if trigger_type == "annual_date":
            month = rule.get("month")
            day = rule.get("day")

            if not month or not day:
                continue

            course_code = rule.get("required_course")
            print("course_code", course_code)
            if course_code:
                for course in course_code:
                    print("course", course)
                    if course not in user["completedCourses"]:
                        print("continue")
                        continue

            current_year = today.year
            due_date = date(current_year, month, day)

            show_days = rule.get("show_days_before") or 0
            show_date = due_date - timedelta(days=show_days)

            if today >= show_date and today <= due_date:
                should_trigger = True

        # if the notification's due date is based on graduation target term (e.g. form due term before target graduation term)
        elif trigger_type == "graduation_based":

            target_term = user.get("graduationTarget")
            term_offset = rule.get("term_offset")

            if not target_term or term_offset is None:
                continue

            course_code = rule.get("required_course")
            print("course_code", course_code)
            if course_code:
                for course in course_code:
                    print("course", course)
                    if course not in user["completedCourses"]:
                        print("continue")
                        continue

            calculated_term = apply_term_offset(target_term, term_offset)

            current_term = get_current_term()
            

            if current_term == calculated_term:
                should_trigger = True
       
        # if the notification's due date is based on program start term (e.g. form due during program start term)
        elif trigger_type == "program_start_based":
            start_term = user.get("startTerm")
            term_offset = rule.get("term_offset")

            if not start_term or term_offset is None:
                continue

            course_code = rule.get("required_course")
            print("course_code", course_code)
            if course_code:
                for course in course_code:
                    print("course", course)
                    if course not in user["completedCourses"]:
                        print("continue")
                        continue

            calculated_term = apply_term_offset(start_term, term_offset)
            current_term = get_current_term()


            if current_term == calculated_term:
                should_trigger = True

        # Create notification
        if should_trigger and not notification_exists(user["id"], rule["id"]):
            create_notification(user["id"], rule)

# Get active notifications (Unread)
def get_active_notifications(user_id):
    response = (
        supabase
        .table("Notifications")
        .select("*, NotificationRules(message)")
        .eq("userId", user_id)
        .eq("read", False)
        .execute()
    )

    return response.data

# Mark notification as read
def mark_as_read(notification_id):
    supabase.table("Notifications") \
        .update({
            "read": True,
            "read_at": datetime.now().isoformat()
        }).eq("id", notification_id).execute()

# Main entry point
def generate_notifications(user_id):
    user = get_user_profile(user_id)
    evaluate_rules(user)
    return get_active_notifications(user_id)