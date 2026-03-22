from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from database import engine, get_db, init_db
from models import Project, Task, Activity
from schemas import ProjectCreate, ProjectResponse, TaskCreate, TaskResponse, ActivityResponse, StatsResponse
from api import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database
    init_db()
    yield
    # Shutdown


app = FastAPI(
    title="Project Dashboard API",
    description="API for project management dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")

# Mount static files for React frontend - must be after API routes
if os.path.exists("static"):
    app.mount("/app", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
async def root():
    """Root endpoint - redirects to React app"""
    return {"message": "Project Dashboard API", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}
