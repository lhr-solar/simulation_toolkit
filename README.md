# Simulation for 2024-26 Design Cycle

To help predict the effects of certain design decisions.

## How to use git
Git is a version control system that helps track changes to files and collaborate with others. This tutorial covers basic Git operations including cloning, changing branches, pulling updates, committing changes, and pushing to a remote repository.

### 1. Cloning a Repository
To work on an existing repository, you first need to clone it.

```bash
# Syntax
git clone <repository-url>

# Example
git clone https://github.com/lhr-solar/simulation_toolkit.git
# or
git clone git@github.com:lhr-solar/simulation_toolkit.git
```

This will create a local copy of the repository in your current directory.

---

### 2. Changing Branches
Branches allow you to work on different features or fixes simultaneously without affecting the main codebase.

#### List All Branches
```bash
git branch -a
```

#### Switch to a Branch
```bash
# Syntax
git checkout <branch-name>

# Example
git checkout feature-branch
```

#### Create and Switch to a New Branch
```bash
# Syntax
git checkout -b <new-branch-name>

# Example
git checkout -b new-feature
```

---

### 3. Pulling Updates
Before making changes, ensure your local repository is up to date with the remote repository.

#### Pull Updates from the Current Branch
```bash
git pull origin <branch-name>

# Example
git pull origin main
```

This fetches the latest changes from the remote branch and merges them into your current branch.

---

### 4. Making Changes and Committing
Once you make changes to files, you can stage and commit them.

#### Stage Changes
```bash
# Stage specific files
git add <file1> <file2>

# Stage all changes
git add .
```

#### Check Status
```bash
git status
```

#### Commit Changes
```bash
# Syntax
git commit -m "Your commit message"

# Example
git commit -m "Fix bug in data processing script"
```

---

### 5. Pushing to a Remote Repository
After committing changes, push them to the remote repository.

#### Push Changes to a Branch
```bash
# Syntax
git push origin <branch-name>

# Example
git push origin main
```

If you're pushing to a new branch for the first time:
```bash
git push -u origin <branch-name>
```

This sets the upstream branch, so you can use `git push` without specifying the branch name in the future.

---

### 6. Submodules
#### Initializing and Cloning Submodules
After cloning a repository with submodules, you need to initialize and fetch the submodule content.
```bash
git submodule init
git submodule update
```

Alternatively, if you want to clone the repository and its submodules in one step, use the following command:
```bash
git clone --recurse-submodules <repository-url>
```

#### Updating Submodules
To ensure that your submodules are up-to-date with the latest changes from the remote repositories, you can update them with:
```bash
git submodule update --remote
```
This fetches the latest commit from the submodule's remote repository and updates your submodule.

#### Adding a Submodule
To add a new submodule to your repository, use the following command. Replace `<repository-url>` with the URL of the submodule repository and `<path-to-submodule>` with the directory where the submodule will be placed in your project.
```bash
# Syntax
git submodule add <repository-url> <path-to-submodule>

# Example
git submodule add https://github.com/lhr-solar/simulation_toolkit.git simulation_toolkit
```
