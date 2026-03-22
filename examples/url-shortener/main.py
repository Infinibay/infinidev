from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import sqlite3
import string
import random
import time
from collections import defaultdict
import re
from contextlib import asynccontextmanager

# Database initialization
def init_db():
    conn = sqlite3.connect('urls.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS urls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            short_code TEXT UNIQUE NOT NULL,
            original_url TEXT NOT NULL,
            click_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database immediately
init_db()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="URL Shortener", lifespan=lifespan)

# Rate limiting storage
rate_limit_store = defaultdict(list)
RATE_LIMIT = 10  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Generate unique short code
def generate_short_code(length=6):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

# Validate URL
def validate_url(url: str) -> bool:
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$',
        re.IGNORECASE
    )
    return pattern.match(url) is not None

# Check rate limit
def check_rate_limit(ip: str) -> bool:
    now = time.time()
    # Clean old entries
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
    # Check limit
    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        return False
    rate_limit_store[ip].append(now)
    return True

# Get client IP
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URL Shortener</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex; justify-content: center; align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 600px; width: 100%;
        }
        h1 { text-align: center; color: #333; margin-bottom: 30px; font-size: 2rem; }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #555; font-weight: 500; }
        input[type="url"] {
            width: 100%; padding: 12px; border: 2px solid #e0e0e0;
            border-radius: 6px; font-size: 16px; transition: border-color 0.3s;
        }
        input[type="url"]:focus { outline: none; border-color: #667eea; }
        button {
            width: 100%; padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 6px;
            font-size: 16px; font-weight: 600; cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        button:active { transform: translateY(0); }
        .result {
            margin-top: 25px; padding: 20px; background: #f8f9fa;
            border-radius: 6px; display: none;
        }
        .result.success { display: block; border-left: 4px solid #48bb78; }
        .result.error { background: #fee; border-left: 4px solid #e53e3e; display: block; }
        .short-url { font-size: 1.2rem; color: #667eea; font-weight: 600; word-break: break-all; }
        .copy-btn {
            background: #e0e0e0; color: #333; margin-top: 10px;
        }
        .copy-btn:hover { background: #d0d0d0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔗 URL Shortener</h1>
        <form id="shorten-form">
            <div class="form-group">
                <label for="url">Enter your long URL</label>
                <input type="url" id="url" name="url" placeholder="https://example.com/very/long/url" required>
            </div>
            <button type="submit">Shorten URL</button>
        </form>
        <div id="result" class="result"></div>
    </div>
    <script>
        document.getElementById('shorten-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const url = document.getElementById('url').value;
            const resultDiv = document.getElementById('result');

            try {
                const response = await fetch('/shorten', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                const data = await response.json();

                if (response.ok) {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <p>Your shortened URL:</p>
                        <p class="short-url">${window.location.origin}/${data.short_code}</p>
                        <button class="copy-btn" onclick="copyURL('${data.short_code}')">Copy to Clipboard</button>
                    `;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p>❌ Error: ${data.detail}</p>`;
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = `<p>❌ Error: ${error.message}</p>`;
            }
        });

        function copyURL(shortCode) {
            const url = window.location.origin + '/' + shortCode;
            navigator.clipboard.writeText(url).then(() => alert('Copied to clipboard!'));
        }
    </script>
</body>
</html>
'''
    return HTMLResponse(content=html_content)

# Shorten URL endpoint
@app.post("/shorten")
async def shorten_url(request: Request, url_data: dict):
    ip = get_client_ip(request)

    # Check rate limit
    if not check_rate_limit(ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a minute before trying again."
        )

    # Extract and validate URL
    url = url_data.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    if not validate_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL format. URL must start with http:// or https://")

    # Generate unique short code
    short_code = generate_short_code()

    # Store in database
    conn = sqlite3.connect('urls.db')
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO urls (short_code, original_url) VALUES (?, ?)",
            (short_code, url)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # If code already exists, generate a new one
        short_code = generate_short_code()
        cursor.execute(
            "INSERT INTO urls (short_code, original_url) VALUES (?, ?)",
            (short_code, url)
        )
        conn.commit()
    finally:
        conn.close()

    return {"short_code": short_code, "original_url": url}

# Health check - must be before /{code} route
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Stats endpoint - must be before /{code} route
@app.get("/stats/{code}")
async def get_stats(code: str):
    conn = sqlite3.connect('urls.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT original_url, click_count, created_at, last_accessed FROM urls WHERE short_code = ?",
        (code,)
    )
    result = cursor.fetchone()
    conn.close()

    if not result:
        raise HTTPException(status_code=404, detail="Short URL not found")

    original_url, click_count, created_at, last_accessed = result

    return {
        "short_code": code,
        "original_url": original_url,
        "click_count": click_count,
        "created_at": created_at,
        "last_accessed": last_accessed
    }

# Redirect endpoint - must be last (catch-all)
@app.get("/{code}")
async def redirect_url(code: str):
    conn = sqlite3.connect('urls.db')
    cursor = conn.cursor()

    cursor.execute(
        "SELECT original_url FROM urls WHERE short_code = ?",
        (code,)
    )
    result = cursor.fetchone()

    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Short URL not found")

    original_url = result[0]

    # Update click count and last accessed
    cursor.execute(
        "UPDATE urls SET click_count = click_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE short_code = ?",
        (code,)
    )
    conn.commit()
    conn.close()

    return RedirectResponse(url=original_url)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
