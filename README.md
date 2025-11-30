# CS 7643 Final Project: CustomerGPT

# Notes
- The library `torch_directml` is NOT compatible on non-Windows machines; do not install if on Linux os MacOS. Instead of `torch`. 

# Instructions to Run on PACE
- Upload all files to `scratch` directory under a single folder.
- Request a Microsoft VS Code Interactive Session from the Interactive Apps > IDEs > Microsoft VS Code.
    - I usually request a 4-core CPU and the first available 32 GB NVIDIA GPU.
- When the session is granted, start the session and navigate to the project directory. 
- ***YOU ONLY NEED TO DO THIS ONCE:***
    - Run `python -m venv env` to create a virtual environment. This will be saved in your project directory and you can activate it in the future without having to install all the libraries all over again. 
    - Run `pip install -r requirements.txt` to install the libraries necessary.
- ***DO THIS EVERY TIME YOU START A NEW PACE SESSION:***
    - Run `source env/bin/activate` to activate the virtual environment.
