---
title: Markdown Features Demo
date: 2024-01-15
tags: [markdown, tutorial, demo]
template: page.html
---

# Markdown Features Demo

This post demonstrates various markdown features supported by the site generator.

## Code Blocks

### Inline Code

Use `backticks` for inline code like this variable: `const name = "hello";`

### Fenced Code Blocks

```python
def hello_world():
    print("Hello, World!")
    return True

if __name__ == "__main__":
    hello_world()
```

```javascript
const greet = (name) => {
    return `Hello, ${name}!`;
};

console.log(greet("Visitor"));
```

## Tables

| Feature | Status | Notes |
|---------|--------|-------|
| Headers | ✅ | Full support |
| Tables | ✅ | As you see here |
| Code | ✅ | With syntax highlighting |
| Footnotes | ✅ | See below[^1] |

## Footnotes

You can add footnotes like this[^2] for additional context or references.

[^1]: Footnotes are rendered at the bottom of the page.
[^2]: This is the second footnote with more information.

## Blockquotes

> This is a blockquote.
> It can span multiple lines.
> 
> And even include nested elements!

## Lists

### Ordered List

1. First item
2. Second item
3. Third item

### Unordered List

* Apple
* Banana
* Cherry

### Nested Lists

* Category 1
  * Sub-item 1
  * Sub-item 2
* Category 2
  * Sub-item A
  * Sub-item B
