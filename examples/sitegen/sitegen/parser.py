"""Markdown and YAML frontmatter parser."""

import re
from pathlib import Path
from typing import Optional

import markdown
import yaml


class ParsedPage:
    """Represents a parsed markdown page with frontmatter."""
    
    def __init__(self, title: str, date: str, content: str, path: Path, 
                 tags: list[str] = None, template: str = None):
        self.title = title
        self.date = date
        self.content = content
        self.path = path
        self.tags = tags or []
        self.template = template or "page.html"
        self.html = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for template rendering."""
        return {
            "title": self.title,
            "date": self.date,
            "content": self.html,
            "path": self.path,
            "tags": self.tags,
            "template": self.template,
        }


class MarkdownParser:
    """Parser for markdown files with YAML frontmatter support."""

    def parse_directory(self, input_dir: Path) -> list[ParsedPage]:
        """
        Parse all markdown files in a directory recursively.

        Args:
            input_dir: Path to input directory

        Returns:
            List of ParsedPage objects sorted by date (newest first)
        """
        pages = []
        for md_file in input_dir.rglob("*.md"):
            page = parse_page(md_file)
            if page:
                pages.append(page)
        
        # Sort by date (newest first)
        return sorted(pages, key=lambda p: p.date, reverse=True)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.
    
    Args:
        content: Raw markdown content with optional YAML frontmatter
        
    Returns:
        Tuple of (frontmatter dict, remaining content)
    """
    frontmatter = {}
    body = content
    
    # Check for YAML frontmatter (--- delimiter)
    if content.startswith("---"):
        end_marker = content.find("---", 3)
        if end_marker != -1:
            yaml_content = content[4:end_marker].strip()
            try:
                frontmatter = yaml.safe_load(yaml_content) or {}
                body = content[end_marker + 3:].strip()
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse frontmatter: {e}")
                frontmatter = {}
                body = content
    
    return frontmatter, body


def parse_markdown(content: str, extensions: list[str] = None) -> str:
    """Convert markdown content to HTML."""
    md_extensions = extensions or [
        "fenced_code",
        "tables",
        "footnotes",
        "toc",
        "nl2br",
        "codehilite",
    ]
    
    md = markdown.Markdown(extensions=md_extensions)
    return md.convert(content)


def parse_page(file_path: Path) -> Optional[ParsedPage]:
    """Parse a markdown file into a ParsedPage object."""
    try:
        content = file_path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)
        
        # Ensure required fields
        title = frontmatter.get("title", file_path.stem)
        date = frontmatter.get("date", "1970-01-01")
        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        template = frontmatter.get("template", "page.html")
        
        # Parse markdown to HTML
        html = parse_markdown(body)
        
        page = ParsedPage(
            title=title,
            date=date,
            content=body,
            path=file_path,
            tags=tags,
            template=template,
        )
        page.html = html
        
        return page
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
