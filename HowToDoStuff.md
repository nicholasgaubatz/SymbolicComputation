### How to create a new Python repository or package stuff

Create repository on GitHub and clone.
> git clone [password-protected SSH key]

Initialize poetry.
> cd [repository_path]
> poetry init

Create a license: MIT is good:
- On "License []" prompt, put MIT
- Go to https://choosealicense.com/licenses/mit/ and copy
- Paste into LICENSE file in root of directory

Create a virtual environment.
> poetry env use /usr/bin/python3
If a .venv directory doesn't show up, need to find global (I think) poetry venv and delete it, then do other stuff that I may write here later.

Install ruff and make sure it works.
> source .venv/bin/activate
> pip install ruff
> ruff version
> ruff check

If you want ruff to automatically fix some things, do 
> ruff check --fix
I'm not sure whether I completely trust this, yet. As of now, I'm not using it.
See https://realpython.com/ruff-python/ for reference.

To check a rule, do
> ruff rule [rule]