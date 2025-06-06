# Copilot Instructions for Julia Differentiation & Neural Network Project

## General Guidelines

- Use Julia programming language, targeting the latest stable release.
- Prefer code that is readable and simple, but prioritize performance when possible.
- Use Julia's performance features (type annotations, in-place operations, multi-threading) whenever beneficial.
- Avoid external dependencies; use Julia's standard library features unless absolutely necessary.
- Minimal documentation is sufficient (short comments only when needed).
- Follow the current workspace organization (e.g., autodiff/, neuralnet/ folders).
- Code does not need to be production-grade; exploratory and experimental code is acceptable.

## Coding Style

- Use clear and descriptive variable/function names, but brevity is acceptable.
- Prefer functions over scripts; modularize code when it helps clarity or reuse.
- Use type annotations and in-place operations for performance.
- Use multi-threading or parallelism if it provides clear performance benefits.
- Avoid unnecessary abstraction or over-engineering.

## Variable Naming

- It is acceptable to use mathematical symbols (e.g., x, y, θ, ∇) for variable names, especially when they improve clarity or align with mathematical conventions in differentiation and neural networks.

## Documentation

- Only add docstrings or comments when the code is non-obvious or complex.
- No need for extensive module or function documentation.

## Dependencies

- Avoid external Julia packages unless required for core functionality.
- If a dependency is necessary, prefer those in the Julia standard library.

## File/Folder Structure

- Keep new code and modules organized according to the current workspace structure.
- Place differentiation-related code in `autodiff/` and neural network code in `neuralnet/`.

## Testing

- Tests are optional, but if included, keep them simple and focused on core functionality.

## User Profile & Preferences

- The user is an experienced developer and IT student, but new to Julia and machine learning/AI/neural networks.
- The user may ask for explanations of machine learning, neural network, or mathematical concepts (e.g., automatic differentiation).
- The user may request explanations or code samples from industry-standard libraries (e.g., Flux.jl) to better understand neural networks and inform their own implementations.
- When providing explanations, prioritize clarity and educational value, using concise examples and analogies where helpful.
- When showing code from external libraries, clearly indicate it is for reference and not for direct use in the project.

---

These rules are intended for Copilot and other AI coding assistants to guide code generation for this project.
