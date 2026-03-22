from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import List, Optional

from database import get_db
from models import Project, Task, Activity, TaskStatus, TaskPriority
from schemas import (
    ProjectCreate, ProjectResponse,
    TaskCreate, TaskUpdate, TaskResponse,
    ActivityResponse, StatsResponse
)

router = APIRouter()


# ============== PROJECT ENDPOINTS ==============

@router.get("/projects", response_model=List[ProjectResponse])
def get_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all projects"""
    projects = db.query(Project).offset(skip).limit(limit).all()
    return projects


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get a specific project by ID"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project"""
    db_project = Project(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


@router.put("/projects/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, project: ProjectCreate, db: Session = Depends(get_db)):
    """Update a project"""
    db_project = db.query(Project).filter(Project.id == project_id).first()
    if not db_project:
        raise HTTPException(status_code=404, detail="Project not found")

    for field, value in project.model_dump().items():
        setattr(db_project, field, value)

    db.commit()
    db.refresh(db_project)
    return db_project


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete associated tasks first
    db.query(Task).filter(Task.project_id == project_id).delete()
    db.query(Activity).filter(Activity.project_id == project_id).delete()
    db.delete(project)
    db.commit()
    return None


# ============== TASK ENDPOINTS ==============

@router.get("/tasks", response_model=List[TaskResponse])
def get_tasks(
    project_id: Optional[int] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all tasks with optional filters"""
    query = db.query(Task)

    if project_id:
        query = query.filter(Task.project_id == project_id)
    if status:
        query = query.filter(Task.status.ilike(status))
    if priority:
        query = query.filter(Task.priority.ilike(priority))

    tasks = query.offset(skip).limit(limit).all()
    return tasks


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: int, db: Session = Depends(get_db)):
    """Get a specific task by ID"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    """Create a new task"""
    # Verify project exists
    project = db.query(Project).filter(Project.id == task.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    db_task = Task(**task.model_dump())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)

    # Log activity
    activity = Activity(
        project_id=task.project_id,
        task_id=db_task.id,
        action="created",
        details=f"Task '{db_task.title}' was created",
    )
    db.add(activity)
    db.commit()

    return db_task


@router.put("/tasks/{task_id}", response_model=TaskResponse)
def update_task(task_id: int, task_update: TaskUpdate, db: Session = Depends(get_db)):
    """Update a task"""
    db_task = db.query(Task).filter(Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Track status changes for activity logging
    old_status = db_task.status
    new_status = task_update.status if task_update.status else old_status
    old_title = db_task.title

    update_data = task_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_task, field, value)

    db.commit()
    db.refresh(db_task)

    # Log activity for status changes
    if old_status != new_status:
        activity = Activity(
            project_id=db_task.project_id,
            task_id=task_id,
            action="status_changed",
            details=f"Task '{db_task.title}' moved from {old_status} to {new_status}",
        )
        db.add(activity)
    # Log activity for title changes
    elif old_title != task_update.title:
        activity = Activity(
            project_id=db_task.project_id,
            task_id=task_id,
            action="updated",
            details=f"Task title changed from '{old_title}' to '{db_task.title}'",
        )
        db.add(activity)

    db.commit()
    return db_task


@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    """Delete a task"""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Log activity before deletion
    activity = Activity(
        project_id=task.project_id,
        task_id=task_id,
        action="deleted",
        details=f"Task '{task.title}' was deleted",
    )
    db.add(activity)

    db.delete(task)
    db.commit()
    return None


# ============== ACTIVITY ENDPOINTS ==============

@router.get("/activity", response_model=List[ActivityResponse])
def get_activity(
    project_id: Optional[int] = None,
    limit: int = 50,
    days: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get recent activity feed"""
    query = db.query(Activity)

    if project_id:
        query = query.filter(Activity.project_id == project_id)

    if days:
        cutoff_date = datetime.now() - timedelta(days=days)
        query = query.filter(Activity.timestamp >= cutoff_date)

    activities = query.order_by(desc(Activity.timestamp)).limit(limit).all()
    return activities


# ============== STATS ENDPOINTS ==============

@router.get("/stats", response_model=StatsResponse)
def get_stats(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Get dashboard statistics"""
    query = db.query(Task)

    if project_id:
        query = query.filter(Task.project_id == project_id)

    # Basic counts
    total_tasks = query.count()
    completed_tasks = query.filter(Task.status.ilike("done")).count()
    in_progress_tasks = query.filter(Task.status.ilike("in progress")).count()

    # Overdue tasks (due_date < now and status != done)
    now = datetime.now()
    overdue_tasks = query.filter(
        Task.due_date < now,
        Task.status.ilike("done").neg()
    ).count()

    # By priority
    priority_counts = db.query(
        Task.priority,
        func.count(Task.id)
    ).group_by(Task.priority).all()
    by_priority = {p: c for p, c in priority_counts}

    # By status
    status_counts = db.query(
        Task.status,
        func.count(Task.id)
    ).group_by(Task.status).all()
    by_status = {s: c for s, c in status_counts}

    # Completion rate per project (computed in Python for SQLite compatibility)
    projects = db.query(Project).all()
    completion_rate_per_project = {}
    for project in projects:
        project_tasks = db.query(Task).filter(Task.project_id == project.id).all()
        total = len(project_tasks)
        completed = len([t for t in project_tasks if t.status.lower() == "done"])
        rate = (completed / total * 100) if total > 0 else 0
        completion_rate_per_project[project.name] = {
            "total": total,
            "completed": completed,
            "rate": round(rate, 1)
        }

    return StatsResponse(
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        overdue_tasks=overdue_tasks,
        in_progress_tasks=in_progress_tasks,
        by_priority=by_priority,
        by_status=by_status,
        completion_rate_per_project=completion_rate_per_project
    )


@router.get("/stats/tasks-by-status", response_model=dict)
def get_tasks_by_status(db: Session = Depends(get_db)):
    """Get task counts grouped by status for charts"""
    status_counts = db.query(
        Task.status,
        func.count(Task.id)
    ).group_by(Task.status).all()

    return {status: count for status, count in status_counts}
