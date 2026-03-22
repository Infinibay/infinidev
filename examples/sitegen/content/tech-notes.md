---
title: Tech Notes
author: Developer
date: 2024-01-15
tags:
  - tech
  - programming
template: page.html
---

# Technical Notes

## Code Examples

Here's a Python example with syntax highlighting:

```python
def fibonacci(n):
    """Generate fibonacci sequence."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Print first 10 fibonacci numbers
for num in fibonacci(10):
    print(num)
```

And a JavaScript example:

```javascript
const greeting = (name) => {
    return `Hello, ${name}!`;
};

console.log(greeting('World'));
```

## Footnotes Example

This is a sentence with a footnote[^1].

Another sentence with a second footnote[^2].

[^1]: This is the first footnote.
[^2]: This is the second footnote with more information.

## Comparison Table

| Feature | Status | Priority |
|---------|--------|----------|
| Syntax Highlighting | ✅ Done | High |
| Tables | ✅ Done | Medium |
| Footnotes | ✅ Done | Low |
| Math | 🔄 In Progress | High |
