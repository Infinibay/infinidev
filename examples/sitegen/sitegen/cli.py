"""Command-line interface for the static site generator."""

import click
from pathlib import Path
from .parser import MarkdownParser
from .renderer import HTMLRenderer
from .utils import copy_static_files


@click.group()
def cli():
    """A minimal static site generator for Markdown files."""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True, file_okay=False),
    default="content",
    help="Input directory containing Markdown files",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="dist",
    help="Output directory for generated HTML",
)
def build(input, output):
    """Build the static site from Markdown files."""
    input_dir = Path(input)
    output_dir = Path(output)

    # Parse all markdown files
    parser = MarkdownParser()
    posts = parser.parse_directory(input_dir)

    # Render HTML
    renderer = HTMLRenderer()
    renderer.render_all(posts, output_dir)

    # Copy static files
    copy_static_files(input_dir / "static", output_dir / "static")

    click.echo(f"✓ Built {len(posts)} pages to {output_dir}")


if __name__ == "__main__":
    cli()
