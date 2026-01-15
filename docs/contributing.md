# :material-hand-heart: Contributing to Lumen

We welcome contributions from the community! Here's how to help.

---

## Ways to Contribute

### 🐛 Report Bugs
Found an issue? [Open a GitHub issue](https://github.com/holoviz/lumen/issues):

- Be specific about what failed
- Include traceback/error message
- Provide minimal reproducible example
- Your environment (OS, Python version, etc.)

### 📚 Improve Documentation
Help others learn Lumen:

- Fix typos and clarify language
- Add examples and use cases
- Improve existing guides
- Submit new tutorials

### ✨ Suggest Features
Have an idea? [Open a GitHub issue](https://github.com/holoviz/lumen/issues):

- Explain the use case
- Describe what you want to build
- Vote on existing feature requests

### 🔧 Contribute Code
- Bug fixes welcome
- New features via discussion first
- Performance improvements
- Tests and test improvements

### 💬 Help in Community
- Answer questions on [Discourse](https://discourse.holoviz.org/c/lumen/14)
- Share examples and projects
- Participate in discussions

---

## Getting Started with Code

### 1. Fork & Clone
```bash
# Visit and fork the repo on GitHub
[https://github.com/holoviz/lumen](https://github.com/holoviz/lumen)
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/lumen.git
cd lumen
```

### 2. Create Environment
```bash
# Using pixi (recommended)
pixi install

# Or using pip
pip install -e ".[tests]"
```

#### Pixi Environments

Lumen uses [pixi](https://pixi.sh) for development. Key environments:

- `test-312` / `test-313`: Full test suite with AI and SQL dependencies
- `test-core`: Minimal test environment
- `docs`: Documentation building
- `lint`: Code linting with pre-commit

### 3. Make Your Changes
```bash
# Create a branch
git checkout -b fix/issue-name

# Make changes, test, commit
git add .
git commit -m "Fix: clear description of change"
```

### 4. Run Tests
```bash
# Run test suite with pixi
pixi run -e test-312 test-unit
```

### 5. Lint Your Code
```bash
# Run linting
pixi run -e lint lint
```

### 6. Build & Preview Docs
```bash
# Serve docs locally with live reload
pixi run -e docs docs-serve

# Build docs
pixi run -e docs docs-build
```

### 7. Submit PR

- Push to your fork
- Open a Pull Request
- Link any related issues
- Describe your changes

---

## Contribution Guidelines

### Code Style
- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Add docstrings to functions/classes

### Testing
- Add tests for new features
- Ensure all tests pass
- Aim for >80% coverage

### Documentation
- Update docs for new features
- Add docstrings
- Include examples

### Commit Messages
- Clear, descriptive messages
- Reference issues: "Fixes #123"
- Use present tense: "Add feature" not "Added feature"

---

## Need Help?

- **Forum:** [Discourse](https://discourse.holoviz.org/c/lumen/)
- **Chat:** [Discord](https://discord.com/invite/rb6gPXbdAr)

---

## Code of Conduct

We're committed to providing a welcoming and inclusive environment.
By participating, you agree to uphold the [HoloViz Code of Conduct](https://holoviz.org/code_of_conduct.html).

---

## Thank You!

Every contribution helps Lumen grow. Thank you for your help! 🙌
