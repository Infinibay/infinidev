"""HTML renderer using Jinja2 templates."""

from pathlib import Path
from typing import List
from jinja2 import Environment, FileSystemLoader, BaseLoader
from .parser import ParsedPage


# Default templates with clean CSS
DEFAULT_BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}{{ suffix }}</title>
    <style>
        :root {
            --bg-color: #fafafa;
            --text-color: #333;
            --link-color: #0366d6;
            --accent-color: #24292e;
            --border-color: #eaecef;
            --code-bg: #f6f8fa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--bg-color);
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }
        header {
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }
        header a {
            color: var(--accent-color);
            text-decoration: none;
            font-size: 1.5rem;
            font-weight: bold;
        }
        header a:hover { color: var(--link-color); }
        nav { margin-top: 0.5rem; }
        nav a {
            margin-right: 1rem;
            color: var(--link-color);
            text-decoration: none;
        }
        nav a:hover { text-decoration: underline; }
        main { min-height: 50vh; }
        article {
            background: white;
            padding: 2rem;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        h1 { font-size: 2.5rem; margin-bottom: 0.5rem; color: var(--accent-color); }
        h2 { font-size: 2rem; margin: 1.5rem 0 0.5rem; }
        h3 { font-size: 1.5rem; margin: 1.2rem 0 0.5rem; }
        p { margin: 1rem 0; }
        a { color: var(--link-color); }
        a:hover { text-decoration: underline; }
        code { 
            background: var(--code-bg); 
            padding: 0.2rem 0.4rem; 
            border-radius: 3px; 
            font-size: 0.9em;
        }
        pre { 
            background: var(--code-bg); 
            padding: 1rem; 
            border-radius: 6px; 
            overflow-x: auto; 
            margin: 1rem 0;
        }
        pre code { background: none; padding: 0; }
        table { 
            border-collapse: collapse; 
            width: 100%; 
            margin: 1rem 0;
        }
        th, td { 
            border: 1px solid var(--border-color); 
            padding: 0.75rem; 
            text-align: left;
        }
        th { background: var(--code-bg); }
        .post-header { margin-bottom: 1.5rem; }
        .post-date { 
            color: #666; 
            font-size: 0.9rem;
        }
        .post-tags { margin-top: 0.5rem; }
        .tag { 
            display: inline-block;
            background: var(--code-bg);
            color: var(--link-color);
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }
        .post-list { list-style: none; }
        .post-item { 
            margin-bottom: 1.5rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        .post-item:last-child { border-bottom: none; }
        .post-item h3 { font-size: 1.5rem; margin: 0 0 0.5rem; }
        .post-item a { text-decoration: none; }
        .post-item a:hover { text-decoration: underline; }
        footer {
            text-align: center;
            padding: 2rem;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <a href="/">{{ site_title }}</a>
            <nav>{{ navigation() }}</nav>
        </header>
        <main>
            {{ content|safe }}
        </main>
        <footer>Generated with sitegen</footer>
    </div>
</body>
</html>
"""

DEFAULT_PAGE_TEMPLATE = """
<article>
    <header class="post-header">
        <h1>{{ page.title }}</h1>
        <p class="post-date">{{ page.date }}</p>
        {% if page.tags %}
        <div class="post-tags">
            {% for tag in page.tags %}
            <span class="tag">{{ tag }}</span>
            {% endfor %}
        </div>
        {% endif %}
    </header>
    <div class="post-content">
        {{ page.html|safe }}
    </div>
</article>
"""

DEFAULT_INDEX_TEMPLATE = """
<div class="index-page">
    <h1>{{ site_title }}</h1>
    <ul class="post-list">
        {% for page in pages %}
        <li class="post-item">
            <h3><a href="{{ page.slug }}/">{{ page.title }}</a></h3>
            <p class="post-date">{{ page.date }}</p>
            {% if page.tags %}
            <div class="post-tags">
                {% for tag in page.tags %}
                <span class="tag">{{ tag }}</span>
                {% endfor %}
            </div>
            {% endif %}
        </li>
        {% endfor %}
    </ul>
</div>
"""


class TemplateLoader(BaseLoader):
    """Custom Jinja2 loader using default templates."""

    def __init__(self, templates: dict = None):
        self.templates = templates or {
            "base.html": DEFAULT_BASE_TEMPLATE,
            "page.html": DEFAULT_PAGE_TEMPLATE,
            "index.html": DEFAULT_INDEX_TEMPLATE,
        }

    def get_source(self, environment, template_name):
        if template_name in self.templates:
            return self.templates[template_name], None, None
        raise ValueError(f"Template {template_name} not found")


class HTMLRenderer:
    """Renders HTML pages using Jinja2 templates."""

    def __init__(self, templates: dict = None):
        """Initialize renderer with custom templates or defaults."""
        self.env = Environment(loader=TemplateLoader(templates), autoescape=True)
        self.env.filters['slugify'] = self.slugify_filter

    @staticmethod
    def slugify_filter(value: str) -> str:
        """Convert string to URL-friendly slug."""
        import re
        value = value.lower().strip()
        value = re.sub(r"[^a-z0-9\s-]", "", value)
        value = re.sub(r"\s+", "-", value)
        return value

    def render_page(self, page: ParsedPage) -> str:
        """Render a single page."""
        page_content = self.env.get_template("page.html").render(page=page)
        return self.env.get_template("base.html").render(
            title=page.title,
            suffix=" - My Site",
            content=page_content,
            site_title="My Site",
            navigation=self._render_navigation,
        )

    def render_index(self, pages: List[ParsedPage]) -> str:
        """Render the index page listing all posts."""
        # Sort pages by date (newest first)
        sorted_pages = sorted(pages, key=lambda p: p.date, reverse=True)
        index_content = self.env.get_template("index.html").render(
            pages=sorted_pages,
            site_title="My Site",
        )
        return self.env.get_template("base.html").render(
            title="Home",
            suffix=" - My Site",
            content=index_content,
            site_title="My Site",
            navigation=self._render_navigation,
        )

    def _render_navigation(self) -> str:
        """Generate navigation HTML."""
        return '<a href="/">Home</a>'

    def render_all(self, pages: List[ParsedPage], output_dir: Path) -> None:
        """Render all pages and write to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Render index page
        index_html = self.render_index(pages)
        index_path = output_dir / "index.html"
        index_path.write_text(index_html, encoding="utf-8")

        # Render each page
        for page in pages:
            page_html = self.render_page(page)
            # Create output path based on slug
            slug = self.slugify_filter(page.title)
            page_dir = output_dir / slug
            page_dir.mkdir(parents=True, exist_ok=True)
            (page_dir / "index.html").write_text(page_html, encoding="utf-8")

    def save_templates(self, templates_dir: Path) -> None:
        """Save default templates to a directory."""
        templates_dir.mkdir(parents=True, exist_ok=True)
        for name, content in self.env.loader.templates.items():
            (templates_dir / name).write_text(content, encoding="utf-8")
