// API base URL
const API_BASE = '';

// Utility functions
const formatDate = (dateString) => {
  if (!dateString) return '';
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return formatDate(timestamp);
};

const getInitials = (name) => {
  if (!name) return '?';
  return name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
};

const getPriorityColor = (priority) => {
  const colors = {
    'High': 'bg-red-100 text-red-800 border-red-300',
    'Medium': 'bg-yellow-100 text-yellow-800 border-yellow-300',
    'Low': 'bg-green-100 text-green-800 border-green-300'
  };
  return colors[priority] || 'bg-gray-100 text-gray-800 border-gray-300';
};

const getStatusColor = (status) => {
  const colors = {
    'Backlog': 'bg-slate-100 border-slate-300',
    'To Do': 'bg-blue-100 border-blue-300',
    'In Progress': 'bg-purple-100 border-purple-300',
    'Review': 'bg-orange-100 border-orange-300',
    'Done': 'bg-green-100 border-green-300'
  };
  return colors[status] || 'bg-gray-100 border-gray-300';
};

// Icons as SVG components
const DashboardIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
  </svg>
);

const KanbanIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
  </svg>
);

const ProjectsIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
  </svg>
);

const TaskIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const ClockIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const ExclamationIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
  </svg>
);

// Stat Card Component
const StatCard = ({ title, value, icon, color }) => (
  <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm text-gray-500 mb-1">{title}</p>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
      </div>
      <div className={`p-3 rounded-lg ${color}`}>{icon}</div>
    </div>
  </div>
);

// Activity Item Component
const ActivityItem = ({ activity }) => (
  <div className="flex items-start space-x-3 py-3 border-b border-gray-100 last:border-0">
    <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
    <div className="flex-1 min-w-0">
      <p className="text-sm text-gray-800">{activity.details}</p>
      <p className="text-xs text-gray-400 mt-1">{formatTimestamp(activity.timestamp)}</p>
    </div>
  </div>
);

// Task Card Component (for Kanban)
const TaskCard = ({ task, onDragStart }) => (
  <div
    draggable
    onDragStart={(e) => onDragStart(e, task.id)}
    className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 cursor-move hover:shadow-md transition-all"
  >
    <div className="flex items-start justify-between mb-2">
      <h4 className="text-sm font-medium text-gray-900 line-clamp-2">{task.title}</h4>
    </div>
    {task.description && (
      <p className="text-xs text-gray-500 mb-3 line-clamp-2">{task.description}</p>
    )}
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-2">
        <span className={`text-xs px-2 py-1 rounded-full border ${getPriorityColor(task.priority)}`}>
          {task.priority}
        </span>
        {task.due_date && (
          <span className={`text-xs flex items-center ${new Date(task.due_date) < new Date() && task.status !== 'Done' ? 'text-red-500' : 'text-gray-400'}`}>
            <ClockIcon />
            <span className="ml-1">{formatDate(task.due_date)}</span>
          </span>
        )}
      </div>
      {task.assignee && (
        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-xs text-white font-medium">
          {getInitials(task.assignee)}
        </div>
      )}
    </div>
  </div>
);

// Project Card Component
const ProjectCard = ({ project, taskCounts }) => {
  const total = taskCounts[project.id]?.total || 0;
  const completed = taskCounts[project.id]?.completed || 0;
  const progress = total > 0 ? (completed / total) * 100 : 0;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: project.color }}></div>
          <h3 className="text-lg font-semibold text-gray-900">{project.name}</h3>
        </div>
        <span className="text-sm text-gray-500">{total} tasks</span>
      </div>
      <p className="text-sm text-gray-500 mb-4 line-clamp-2">{project.description}</p>
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Progress</span>
          <span className="text-gray-700 font-medium">{Math.round(progress)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="h-2 rounded-full transition-all" 
            style={{ width: `${progress}%`, backgroundColor: project.color }}
          ></div>
        </div>
      </div>
    </div>
  );
};

// Dashboard Page
const DashboardPage = ({ stats, activities, setStats, setActivities }) => {
  const [chartData, setChartData] = React.useState([]);

  const fetchData = async () => {
    try {
      const [statsRes, activitiesRes] = await Promise.all([
        fetch(`${API_BASE}/stats`),
        fetch(`${API_BASE}/activity?limit=10`)
      ]);
      if (statsRes.ok) setStats(await statsRes.json());
      if (activitiesRes.ok) setActivities(await activitiesRes.json());
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    }
  };

  React.useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  React.useEffect(() => {
    if (stats && stats.by_status) {
      const statuses = ['Backlog', 'To Do', 'In Progress', 'Review', 'Done'];
      const data = statuses.map(status => stats.by_status[status] || 0);
      setChartData(data);
    }
  }, [stats]);

  React.useEffect(() => {
    if (chartData.length > 0) {
      const ctx = document.getElementById('statusChart');
      if (ctx) {
        new Chart(ctx, {
          type: 'doughnut',
          data: {
            labels: ['Backlog', 'To Do', 'In Progress', 'Review', 'Done'],
            datasets: [{
              data: chartData,
              backgroundColor: ['#64748b', '#3b82f6', '#a855f7', '#f97316', '#22c55e'],
              borderWidth: 0
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'bottom',
                labels: {
                  padding: 20,
                  font: { size: 11 }
                }
              }
            },
            cutout: '60%'
          }
        });
      }
    }
  }, [chartData]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Dashboard</h2>
        <p className="text-gray-500">Overview of your projects and tasks</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total Tasks"
          value={stats?.total_tasks || 0}
          icon={<TaskIcon />}
          color="bg-blue-100 text-blue-600"
        />
        <StatCard
          title="Completed"
          value={stats?.completed_tasks || 0}
          icon={<CheckCircleIcon />}
          color="bg-green-100 text-green-600"
        />
        <StatCard
          title="In Progress"
          value={stats?.in_progress_tasks || 0}
          icon={<ClockIcon />}
          color="bg-purple-100 text-purple-600"
        />
        <StatCard
          title="Overdue"
          value={stats?.overdue_tasks || 0}
          icon={<ExclamationIcon />}
          color="bg-red-100 text-red-600"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Tasks by Status</h3>
          <div className="h-64">
            <canvas id="statusChart"></canvas>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
          <div className="space-y-1">
            {activities?.slice(0, 8).map(activity => (
              <ActivityItem key={activity.id} activity={activity} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// Kanban Board Page
const KanbanBoardPage = ({ tasks, setTasks }) => {
  const statuses = ['Backlog', 'To Do', 'In Progress', 'Review', 'Done'];

  const fetchTasks = async () => {
    try {
      const res = await fetch(`${API_BASE}/tasks`);
      if (res.ok) setTasks(await res.json());
    } catch (error) {
      console.error('Failed to fetch tasks:', error);
    }
  };

  React.useEffect(() => {
    fetchTasks();
  }, []);

  const handleDragStart = (e, taskId) => {
    e.dataTransfer.setData('taskId', taskId.toString());
  };

  const handleDrop = async (e, status) => {
    e.preventDefault();
    const taskId = parseInt(e.dataTransfer.getData('taskId'));
    
    try {
      const res = await fetch(`${API_BASE}/tasks/${taskId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      
      if (res.ok) {
        const updatedTask = await res.json();
        setTasks(prev => prev.map(t => t.id === taskId ? updatedTask : t));
      }
    } catch (error) {
      console.error('Failed to update task:', error);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const getTasksByStatus = (status) => {
    return tasks.filter(t => t.status === status);
  };

  return (
    <div className="h-[calc(100vh-140px)] overflow-x-auto">
      <div className="flex space-x-4 min-w-max pb-4">
        {statuses.map(status => (
          <div
            key={status}
            onDrop={(e) => handleDrop(e, status)}
            onDragOver={handleDragOver}
            className="w-80 bg-gray-50 rounded-xl p-4"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-gray-700">{status}</h3>
              <span className="text-sm text-gray-400">
                {getTasksByStatus(status).length}
              </span>
            </div>
            <div className="space-y-3">
              {getTasksByStatus(status).map(task => (
                <TaskCard
                  key={task.id}
                  task={task}
                  onDragStart={handleDragStart}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Projects Page
const ProjectsPage = ({ projects, taskCounts }) => {
  const fetchProjects = async () => {
    try {
      const res = await fetch(`${API_BASE}/projects`);
      if (res.ok) {
        const projectsData = await res.json();
        setProjects(projectsData);
        
        const counts = {};
        for (const project of projectsData) {
          const tasksRes = await fetch(`${API_BASE}/tasks?project_id=${project.id}`);
          if (tasksRes.ok) {
            const tasks = await tasksRes.json();
            const completed = tasks.filter(t => t.status === 'Done').length;
            counts[project.id] = { total: tasks.length, completed };
          }
        }
        setTaskCounts(counts);
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error);
    }
  };

  const [projectsData, setProjects] = React.useState([]);
  
  React.useEffect(() => {
    fetchProjects();
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Projects</h2>
        <p className="text-gray-500">Manage your projects</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projectsData.map(project => (
          <ProjectCard
            key={project.id}
            project={project}
            taskCounts={taskCounts}
          />
        ))}
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [currentPage, setCurrentPage] = React.useState('dashboard');
  const [stats, setStats] = React.useState(null);
  const [activities, setActivities] = React.useState([]);
  const [tasks, setTasks] = React.useState([]);
  const [projects, setProjects] = React.useState([]);
  const [taskCounts, setTaskCounts] = React.useState({});

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return (
          <DashboardPage
            stats={stats}
            activities={activities}
            setStats={setStats}
            setActivities={setActivities}
          />
        );
      case 'kanban':
        return <KanbanBoardPage tasks={tasks} setTasks={setTasks} />;
      case 'projects':
        return <ProjectsPage projects={projects} taskCounts={taskCounts} />;
      default:
        return <DashboardPage stats={stats} activities={activities} setStats={setStats} setActivities={setActivities} />;
    }
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <DashboardIcon /> },
    { id: 'kanban', label: 'Kanban Board', icon: <KanbanIcon /> },
    { id: 'projects', label: 'Projects', icon: <ProjectsIcon /> }
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-6">
          <h1 className="text-xl font-bold text-gray-900">Project Dashboard</h1>
        </div>
        <nav className="flex-1 px-4 space-y-2">
          {navItems.map(item => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                currentPage === item.id
                  ? 'bg-blue-50 text-blue-600'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {item.icon}
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-white text-sm font-medium">
              US
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900">User</p>
              <p className="text-xs text-gray-500">Admin</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-8">
          {renderPage()}
        </div>
      </main>
    </div>
  );
};

// Render the app
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
