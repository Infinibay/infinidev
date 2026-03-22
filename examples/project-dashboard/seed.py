from database import SessionLocal, init_db
from models import Project, Task, Activity
from datetime import datetime, timedelta
import random

# Sample data
projects_data = [
    {"name": "Website Redesign", "description": "Complete overhaul of company website with modern UI/UX", "color": "#3B82F6"},
    {"name": "Mobile App", "description": "iOS and Android application for customer engagement", "color": "#10B981"},
    {"name": "Data Pipeline", "description": "ETL pipeline for analytics and reporting", "color": "#8B5CF6"},
]

assignees = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

tasks_data = [
    # Website Redesign tasks (project_id = 1)
    {"title": "Design homepage mockup", "project_id": 1, "status": "Done", "priority": "High", "assignee": "Alice", "due_date": "2024-01-10"},
    {"title": "Implement responsive navigation", "project_id": 1, "status": "In Progress", "priority": "High", "assignee": "Bob", "due_date": "2024-01-15"},
    {"title": "Create contact form", "project_id": 1, "status": "To Do", "priority": "Medium", "assignee": "Charlie", "due_date": "2024-01-20"},
    {"title": "Optimize images", "project_id": 1, "status": "Review", "priority": "Low", "assignee": "Diana", "due_date": "2024-01-18"},
    {"title": "Add analytics tracking", "project_id": 1, "status": "To Do", "priority": "Medium", "assignee": "Alice", "due_date": "2024-01-22"},
    {"title": "SEO optimization", "project_id": 1, "status": "Backlog", "priority": "Medium", "assignee": "Eve", "due_date": "2024-01-25"},
    {"title": "Accessibility audit", "project_id": 1, "status": "Backlog", "priority": "High", "assignee": "Bob", "due_date": "2024-01-28"},
    {"title": "Performance testing", "project_id": 1, "status": "To Do", "priority": "High", "assignee": "Charlie", "due_date": "2024-01-30"},
    
    # Mobile App tasks (project_id = 2)
    {"title": "Set up React Native project", "project_id": 2, "status": "Done", "priority": "High", "assignee": "Alice", "due_date": "2024-01-08"},
    {"title": "Implement user authentication", "project_id": 2, "status": "In Progress", "priority": "High", "assignee": "Diana", "due_date": "2024-01-14"},
    {"title": "Build dashboard screen", "project_id": 2, "status": "Review", "priority": "High", "assignee": "Bob", "due_date": "2024-01-16"},
    {"title": "Add push notifications", "project_id": 2, "status": "To Do", "priority": "Medium", "assignee": "Eve", "due_date": "2024-01-21"},
    {"title": "Offline mode support", "project_id": 2, "status": "Backlog", "priority": "Low", "assignee": "Charlie", "due_date": "2024-01-26"},
    {"title": "App store submission", "project_id": 2, "status": "Backlog", "priority": "Medium", "assignee": "Alice", "due_date": "2024-02-01"},
    {"title": "Bug fixes and polish", "project_id": 2, "status": "To Do", "priority": "Medium", "assignee": "Diana", "due_date": "2024-01-24"},
    
    # Data Pipeline tasks (project_id = 3)
    {"title": "Design database schema", "project_id": 3, "status": "Done", "priority": "High", "assignee": "Charlie", "due_date": "2024-01-05"},
    {"title": "Build data extraction module", "project_id": 3, "status": "Done", "priority": "High", "assignee": "Eve", "due_date": "2024-01-12"},
    {"title": "Implement transformation logic", "project_id": 3, "status": "In Progress", "priority": "High", "assignee": "Alice", "due_date": "2024-01-17"},
    {"title": "Set up data loading", "project_id": 3, "status": "To Do", "priority": "High", "assignee": "Bob", "due_date": "2024-01-19"},
    {"title": "Error handling and logging", "project_id": 3, "status": "Review", "priority": "Medium", "assignee": "Diana", "due_date": "2024-01-23"},
    {"title": "Monitoring dashboard", "project_id": 3, "status": "Backlog", "priority": "Medium", "assignee": "Charlie", "due_date": "2024-01-27"},
    {"title": "Documentation", "project_id": 3, "status": "Backlog", "priority": "Low", "assignee": "Eve", "due_date": "2024-01-29"},
    {"title": "Performance optimization", "project_id": 3, "status": "To Do", "priority": "Medium", "assignee": "Alice", "due_date": "2024-01-31"},
]

def seed_database():
    """Seed the database with sample data"""
    init_db()
    db = SessionLocal()
    
    try:
        # Create projects
        projects = []
        for proj_data in projects_data:
            project = Project(**proj_data)
            db.add(project)
            projects.append(project)
        db.commit()
        # Refresh individual projects to get their IDs
        for project in projects:
            db.refresh(project)
        
        # Create tasks
        for task_data in tasks_data:
            # Parse due_date string to datetime.date object
            if task_data.get('due_date'):
                date_str = task_data['due_date']
                task_data['due_date'] = datetime.strptime(date_str, '%Y-%m-%d').date()
            task = Task(**task_data)
            db.add(task)
        db.commit()
        
        # Create some initial activity records
        tasks = db.query(Task).all()
        for task in tasks[:5]:  # Add activity for first 5 tasks
            activity = Activity(
                project_id=task.project_id,
                task_id=task.id,
                action="created",
                details=f"Task '{task.title}' was created",
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 48))
            )
            db.add(activity)
        db.commit()
        
        print(f"✓ Seeded {len(projects)} projects")
        print(f"✓ Seeded {len(tasks_data)} tasks")
        print(f"✓ Seeded initial activity records")
        
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
