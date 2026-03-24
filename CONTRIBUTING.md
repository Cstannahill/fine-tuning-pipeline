# Contributing to Unsloth Fine-tuning Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

---

## Code of Conduct

- Be respectful and inclusive
- Assume good intent in discussions
- Focus on ideas, not people
- Report concerns to maintainers

---

## Getting Started

### Prerequisites

- Python 3.8+
- Git
- An NVIDIA GPU (recommended for testing training code)
- Familiarity with:
  - PyTorch and Transformers
  - YAML configuration
  - Command-line tools

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/fine-tuning.git
cd fine-tuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Create a development branch
git checkout -b feature/your-feature-name
```

---

## Types of Contributions

### 🐛 Bug Reports

Found a bug? Please open an issue with:

1. **Description** – What went wrong?
2. **Reproduction** – Steps to reproduce the issue
3. **Expected behavior** – What should happen?
4. **Actual behavior** – What actually happened?
5. **Environment** – OS, Python version, GPU, CUDA version
6. **Logs** – Relevant error messages or logs
7. **Minimal example** – Smallest code that reproduces the issue

Use the bug report template on GitHub.

### 💡 Feature Requests

Have an idea? Please open an issue with:

1. **Description** – What feature do you want?
2. **Motivation** – Why is this useful?
3. **Use cases** – How would you use it?
4. **Alternatives** – Any alternative approaches?
5. **Additional context** – Links, examples, references

Use the feature request template on GitHub.

### 📝 Documentation

Help improve documentation:

- Fix typos and improve clarity
- Add examples and tutorials
- Expand troubleshooting section
- Add diagrams and visualizations
- Improve code comments

No formal PR process needed for small doc fixes – just submit directly.

### 🔧 Code Contributions

### Areas for Contribution

- **New features** – New training methods, optimizations, integrations
- **Bug fixes** – Resolve open issues
- **Performance** – Optimize memory, speed, or startup time
- **Tests** – Add unit and integration tests
- **Tools** – New utility scripts or integrations
- **Examples** – Training examples, tutorials, recipes

---

## Development Workflow

### 1. Create a Branch

```bash
# For features
git checkout -b feature/feature-name

# For bug fixes
git checkout -b fix/bug-name

# For documentation
git checkout -b docs/topic-name
```

### 2. Make Changes

```bash
# Edit files
# Follow code style (see below)
# Add tests for new functionality
# Update documentation
```

### 3. Code Style

We follow PEP 8 with these tools:

```bash
# Format code
black src/ tools/

# Check style
flake8 src/ --max-line-length=100
mypy src/ --ignore-missing-imports

# Lint
pylint src/
```

**Guidelines:**

- Use type hints for functions: `def process(data: List[str]) -> Dict[str, Any]:`
- Write docstrings for classes and public functions:
  ```python
  def train_model(config: dict) -> Model:
      """Train a model using the provided configuration.
      
      Args:
          config: Configuration dictionary with 'model' and 'training' keys
          
      Returns:
          Trained Model instance
          
      Raises:
          ValueError: If configuration is invalid
      """
  ```
- Keep lines under 100 characters
- Use descriptive variable names
- Add comments for complex logic
- Avoid hardcoded values (use config)

### 4. Write Tests

For new features, add tests:

```bash
# Create test file
# tests/test_new_feature.py

import pytest
from src.new_module import new_function

def test_basic_functionality():
    result = new_function("test_input")
    assert result == "expected_output"

def test_error_handling():
    with pytest.raises(ValueError):
        new_function(None)

# Run tests
pytest tests/test_new_feature.py -v
```

### 5. Update Documentation

- **README.md** – Update if adding user-facing features
- **Docstrings** – Document all public functions and classes
- **Inline comments** – Explain complex logic
- **Examples** – Add usage examples for new features

### 6. Commit Changes

```bash
# Use clear, descriptive commit messages
git add .
git commit -m "feat: add dataset inspector tool

- Automatically detect dataset format (alpaca, sharegpt, text)
- Auto-register datasets in dataset_info.json
- Support JSON, JSONL, CSV, TXT, Parquet formats
- Show preview and inferred configuration"
```

**Commit message format:**

```
<type>: <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Examples:
- `feat: add dataset inspector tool`
- `fix: correct device mapping in multi-GPU setup`
- `docs: improve README with examples`
- `perf: optimize data loading performance`

### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Open a Pull Request on GitHub with:

1. **Description** – What does this PR do?
2. **Motivation** – Why is this change needed?
3. **Changes** – List of key changes
4. **Testing** – How was this tested?
5. **Checklist** – ✓ Tests pass, docs updated, etc.

Use the PR template on GitHub.

---

## Pull Request Process

### Review Process

1. **Automated checks** – Tests, linting, build checks must pass
2. **Code review** – At least one maintainer review required
3. **Address feedback** – Make requested changes
4. **Approval** – PR approved when ready
5. **Merge** – Maintainer merges to main branch

### Expectations

- **Response time** – Expect feedback within 1-2 weeks
- **Revisions** – Be open to feedback and suggestions
- **Scope** – Keep PRs focused (one feature per PR)
- **Size** – Smaller PRs merge faster (< 400 lines preferred)

### Tips for Successful PRs

✅ **Do:**
- Write clear commit messages
- Keep changes focused
- Add tests for new features
- Update documentation
- Follow code style guidelines
- Respond to feedback promptly

❌ **Don't:**
- Mix multiple features in one PR
- Include unrelated changes
- Skip tests
- Ignore code style
- Submit incomplete work

---

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data.py -v

# Run specific test function
pytest tests/test_data.py::test_load_dataset -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

```python
# tests/test_module.py
import pytest
from src.module import function

class TestClass:
    """Test cases for MyClass"""
    
    def setup_method(self):
        """Run before each test"""
        self.test_data = sample_data()
    
    def test_basic_functionality(self):
        """Test basic case"""
        result = function(self.test_data)
        assert result.status == "success"
    
    def test_edge_case(self):
        """Test edge case"""
        result = function({})
        assert result is None
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            function(None)

def test_integration():
    """Integration test across modules"""
    # Test interaction between modules
    pass
```

### Coverage

Aim for:
- **Core logic** – 100% coverage
- **Utilities** – 90%+ coverage
- **Integration** – Key paths covered
- **Overall** – 85%+ coverage target

---

## Documentation Standards

### Code Comments

```python
def train(config: dict) -> Model:
    """Train a model based on configuration.
    
    This function initializes the model, loads data, and runs training
    with the provided hyperparameters.
    
    Args:
        config: Training configuration dictionary with keys:
            - model: Model configuration (name, type, etc.)
            - training: Training parameters (epochs, lr, etc.)
            - dataset: Dataset configuration
    
    Returns:
        Trained model instance ready for inference
    
    Raises:
        ValueError: If configuration is invalid or missing required keys
        FileNotFoundError: If dataset files don't exist
        RuntimeError: If training fails (GPU error, etc.)
    
    Example:
        >>> config = load_yaml('config.yaml')
        >>> model = train(config)
        >>> predictions = model.predict(data)
    """
    pass
```

### Documentation Files

- **README.md** – Project overview, quick start, basic usage
- **QUICKSTART.md** – 5-minute setup and first run
- **docs/TRAINING_CONFIG.md** – Configuration reference
- **docs/DATASET_REGISTRY.md** – Dataset registry guide
- **RELEASE_NOTES.md** – Version history and changes
- **CONTRIBUTING.md** – This file

---

## Release Process

### For Maintainers

1. Update version number in `src/__init__.py`
2. Update `RELEASE_NOTES.md` with changes
3. Tag release: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions creates release automatically

### Version Numbering

We use semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR** – Breaking changes
- **MINOR** – New features (backward compatible)
- **PATCH** – Bug fixes

---

## Getting Help

### Questions & Discussion

- **GitHub Discussions** – For questions and ideas
- **Issues** – For bugs and feature requests
- **Email** – For security issues (don't use issues!)

### Developer Resources

- **Architecture** – See code structure in README
- **Configuration** – See `docs/TRAINING_CONFIG.md`
- **Examples** – See `examples/` directory
- **API** – See docstrings in source code

---

## Recognition

Contributors are recognized in:

- **GitHub** – Listed as contributor
- **CHANGELOG** – Mentioned in release notes
- **README** – Added to acknowledgments section (if significant)

---

## Questions?

- Contact maintainers: [Email/Contact]
- Open discussion: GitHub Discussions
- Check search: Maybe someone asked already

---

## Thank You! 🙏

Thank you for contributing to Unsloth Fine-tuning Pipeline!

Your efforts help make this project better for everyone.

---

**Happy hacking! 🚀**
