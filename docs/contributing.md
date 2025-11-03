# Contributing to Lumen

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
Have an idea? [Start a discussion](https://github.com/holoviz/lumen/discussions):
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
# Fork the repo on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/lumen.git
cd lumen
```

### 2. Create Environment
```bash
# Using conda (recommended)
conda create -n lumen-dev python=3.11
conda activate lumen-dev
pip install -e ".[all,dev]"

# Or using venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -e ".[all,dev]"
```

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
# Run test suite
pytest tests/

# Run with coverage
pytest --cov=lumen tests/
```

### 5. Submit PR
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

- 💬 [Ask on Discourse](https://discourse.holoviz.org/c/lumen/14)
- 💭 [GitHub Discussions](https://github.com/holoviz/lumen/discussions)
- 📖 [Contributing Guide](https://github.com/holoviz/lumen/blob/main/CONTRIBUTING.md)

---

## Code of Conduct

We're committed to providing a welcoming and inclusive environment.
By participating, you agree to uphold our [Code of Conduct](https://github.com/holoviz/lumen/blob/main/CODE_OF_CONDUCT.md).

---

## Thank You!

Every contribution helps Lumen grow. Thank you for your help! 🙌
