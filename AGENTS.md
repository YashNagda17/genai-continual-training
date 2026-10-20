Follow [Idea.md](/Idea.md), [code-structure.md](/code-structure.md), [ARCHITECTURE.md](/ARCHITECTURE.md), and [SKILLS.md](/SKILLS.md). Keep code simple and use the specified class boundaries.

Use namespace packages and direct module imports. Do not create empty or re-export-only `__init__.py` files; add one only when actual package initialization is explicitly needed. Use native Torch for training, numerical calculations and inference, with GPU detection and CPU fallback. Do not add external inference backends unless requested.

Read [design/README.md](/design/README.md), the affected component contract, and its [paper reference](/Papers/README.md). Change the contract before implementing incompatible behavior.

All model roles are capped at 700M total parameters. [model-weights.md](/design/model-weights.md) defines mandatory resume, bounded checkpoint retention, and model residency.

Append potentially novel algorithmic ideas to [optimizations.md](/optimizations.md), following its six-line template. This is the sole idea log.

Use repository-root local links such as `[scope](/design/README.md)`, never machine-specific paths. Leading `/` means the project root in repository Markdown.
