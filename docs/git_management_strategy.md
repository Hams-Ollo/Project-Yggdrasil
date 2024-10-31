# Git Management Strategy

This document outlines the git workflow and best practices for contributing to this project.

## Branch Strategy

We follow a modified Git Flow strategy with the following main branches:

- `main` - Production-ready code
- `develop` - Main development branch
- Feature branches - For new features/changes
- Hotfix branches - For urgent production fixes

### Branch Naming Convention

- Feature branches: `feature/descriptive-name`
- Hotfix branches: `hotfix/issue-description`
- Release branches: `release/vX.Y.Z`

## Workflow

1. **Starting New Work**
   - Create a new feature branch from `develop`:

   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Making Changes**
   - Make small, focused commits
   - Write clear commit messages:

   ```bash
   git commit -m "feat: add multi-agent chat capability"
   ```

3. **Keeping Up-to-Date**
   - Regularly sync with develop:

   ```bash
   git fetch origin
   git rebase origin/develop
   ```

4. **Submitting Changes**
   - Push your branch to remote:

   ```bash
   git push origin feature/your-feature-name
   ```

   - Create a Pull Request to `develop`
   - Request code review from team members
   - Address review feedback with additional commits
   - Squash commits before merging if needed

5. **Code Review Process**
   - At least one approval required before merging
   - All CI checks must pass
   - No merge conflicts with target branch
   - Code follows project standards

## Commit Message Guidelines

Follow the Conventional Commits specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Adding/modifying tests
- `chore:` - Maintenance tasks

Example:

```bash
git commit -m "feat: add multi-agent chat capability"
```
