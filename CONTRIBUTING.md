# Contributing to MLOps Credit Score Prediction Project

Thank you for considering contributing to this project! This document outlines the guidelines for contributing.

## Development Workflow

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlops-credit-score
```

### 2. Set Up Development Environment
```bash
./scripts/setup_dev.sh
source venv/bin/activate
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Write clean, well-documented code
- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation as needed

### 5. Run Tests
```bash
pytest app/tests/ -v
```

### 6. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

Follow conventional commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test changes
- `refactor:` - Code refactoring
- `ci:` - CI/CD changes

### 7. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## Code Style

### Python
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Type hints are encouraged

### Infrastructure Code
- Use consistent naming conventions
- Comment complex configurations
- Test infrastructure changes in a sandbox first

## Testing Requirements

- All new features must include unit tests
- Maintain test coverage above 80%
- Integration tests for API endpoints
- Infrastructure changes should be tested with `docker-compose up`

## Pull Request Guidelines

1. Keep PRs focused and small
2. Include a clear description of changes
3. Reference any related issues
4. Ensure all tests pass
5. Request review from at least one team member

## Team Responsibilities

### Person 1 - ML/Application Focus
- ML model development and training
- Flask API implementation
- Unit testing
- Docker containerization
- Documentation

### Person 2 - Infrastructure/DevOps Focus
- Jenkins pipeline development
- Kubernetes configuration
- Ansible playbooks
- ELK Stack setup
- Monitoring and alerting

## Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced
- [ ] Performance considerations addressed
- [ ] Configuration is externalized appropriately

## Questions?

If you have questions about contributing, please reach out to the team leads.
