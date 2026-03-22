from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional, List


# Project schemas
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = ""
    color: str = "#3B82F6"


class ProjectCreate(ProjectBase):
    pass


class ProjectResponse(ProjectBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Task schemas
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = ""
    status: str = "backlog"
    priority: str = "medium"
    assignee: Optional[str] = ""
    due_date: Optional[date] = None


class TaskCreate(TaskBase):
    project_id: int


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    assignee: Optional[str] = None
    due_date: Optional[date] = None


class TaskResponse(TaskBase):
    id: int
    project_id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# Activity schemas
class ActivityResponse(BaseModel):
    id: int
    project_id: int
    task_id: Optional[int] = None
    action: str
    details: str
    timestamp: datetime
    
    class Config:
        from_attributes = True


# Stats schemas
class StatsResponse(BaseModel):
    total_tasks: int
    completed_tasks: int
    overdue_tasks: int
    in_progress_tasks: int
    by_priority: dict
    by_status: dict
    completion_rate_per_project: dict
    
    class Config:
        from_attributes = True
