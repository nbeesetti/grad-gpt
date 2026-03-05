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

# Delete or mark as read notifications that no longer should be triggered
def remove_stale_notifications(user):
    user_id = user["id"]
    active_notifications = supabase.table("Notifications") \
        .select("*, NotificationRules(*)") \
        .eq("userId", user_id) \
        .eq("read", False) \
        .execute().data

    completed = user.get("completedCourses") or []

    today = date.today()

    for notif in active_notifications:
        rule = notif["NotificationRules"]
        trigger_type = rule.get("trigger_type")
        stale = False
        notif_type = rule.get("type")

        # Universal stale rule
        if notif_type == "course":
            target_course = rule.get("name")

            # If the course is already completed -> remove notification
            if target_course and target_course in completed:
                stale = True

        # Status cleanup rule
        if trigger_type == "status_based":
            required_status = "Undergraduate"
            current_status = user.get("status")

            if current_status != required_status:
                stale = True

        # Annual date notifications
        elif trigger_type == "annual_date":
            month = rule.get("month")
            day = rule.get("day")
            course_code = rule.get("required_course") or []

            if not all(course in completed for course in course_code):
                stale = True
            elif month and day:
                due_date = date(today.year, month, day)
                show_days = rule.get("show_days_before") or 0
                show_date = due_date - timedelta(days=show_days)
                if not (show_date <= today <= due_date):
                    stale = True
            else:
                stale = True

        # Graduation-based notifications
        elif trigger_type == "graduation_based":
            target_term = user.get("graduationTarget")
            term_offset = rule.get("term_offset")
            course_code = rule.get("required_course") or []

            if not target_term or term_offset is None:
                stale = True
            elif not all(course in completed for course in course_code):
                stale = True
            else:
                calculated_term = apply_term_offset(target_term, term_offset)
                current_term = get_current_term()
                if term_to_number(current_term) < term_to_number(calculated_term):
                    stale = True

        # Program start-based notifications
        elif trigger_type == "program_start_based":
            start_term = user.get("startTerm")
            term_offset = rule.get("term_offset")
            course_code = rule.get("required_course") or []

            if not start_term or term_offset is None:
                stale = True
            elif not all(course in completed for course in course_code):
                stale = True
            else:
                calculated_term = apply_term_offset(start_term, term_offset)
                current_term = get_current_term()
                if term_to_number(current_term) < term_to_number(calculated_term):
                    stale = True

        # Delete if stale
        if stale:
            supabase.table("Notifications").delete().eq("id", notif["id"]).execute()


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

# helper to compare if current term is after term offset for notif
def term_to_number(term_string):
    term, year = term_string.split()
    year = int(year)

    term_map = {
        "Winter": 0,
        "Spring": 1,
        "Summer": 2,
        "Fall": 3,
    }

    return year * 4 + term_map[term]

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

    completed = user.get("completedCourses") or []

    for rule in rules:
        should_trigger = False
        trigger_type = rule.get("trigger_type")

        user_status = user.get("status")

        # for students that are planning on applying to the grad program
        if rule.get("trigger_type") == "status_based":
            required_status = "Undergraduate"  # e.g. "undergrad"

            if user_status == required_status:
                should_trigger = True
            else:
                continue  # Don't create

        # DO NOT CREATE if target course already completed
        target_course = rule.get("name")
        if target_course and target_course in completed:
            continue

        # if the notification's due date is on an annual date (e.g. every year on oct. 10)
        if trigger_type == "annual_date":
            month = rule.get("month")
            day = rule.get("day")

            if not month or not day:
                continue

            course_code = rule.get("required_course")
            if course_code:
                if not all(course in completed for course in course_code):
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
            if course_code:
                 if not all(course in completed for course in course_code):
                    continue 

            calculated_term = apply_term_offset(target_term, term_offset)

            current_term = get_current_term()
            
            if term_to_number(current_term) >= term_to_number(calculated_term):
                should_trigger = True
       
        # if the notification's due date is based on program start term (e.g. form due during program start term)
        elif trigger_type == "program_start_based":
            start_term = user.get("startTerm")
            term_offset = rule.get("term_offset")

            if not start_term or term_offset is None:
                continue

            course_code = rule.get("required_course")
            if course_code:
                if not all(course in completed for course in course_code):
                    continue 

            calculated_term = apply_term_offset(start_term, term_offset)
            current_term = get_current_term()

            if term_to_number(current_term) >= term_to_number(calculated_term):
                should_trigger = True

        # Create notification
        if should_trigger and not notification_exists(user["id"], rule["id"]):
            create_notification(user["id"], rule)

# Get active notifications (Unread)
def get_active_notifications(user_id):
    response = (
        supabase
        .table("Notifications")
        .select("*, NotificationRules(message, month, day, due_date)")
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

    # Remove notifications that are no longer valid
    remove_stale_notifications(user)

    # Then create new ones as usual
    evaluate_rules(user)
    return get_active_notifications(user_id)