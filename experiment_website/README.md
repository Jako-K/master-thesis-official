# Overview

This repo contains:  
1. A self-hosted Python + Flask server for audio-to-image labeling.  
2. A lightweight frontend built only with vanilla JS, HTML, and CSS.  
3. A Fully custom backend: encrypted login, user database, logging, etc. are implemented from scratch.  

[![readme_figure](https://github.com/user-attachments/assets/71d91f82-3f88-4bb2-a6a3-5b2a9beec983)](https://github.com/user-attachments/assets/6e716ae3-2b07-48d1-b78e-4038a3d93bf9)

---

# How to Start the Server

**Using nginx + gunicorn example**
```bash
cd ~/experiment_website
gunicorn --bind 127.0.0.1:5000 main:app --daemon --access-logfile gunicorn_access.log --error-logfile gunicorn_error.log
sudo systemctl start nginx
ps aux | grep gunicorn  # Check it's running correctly
```

**Debug locally**
```bash
python main.py
```

---

# How to Modify the Code for Your Data
Update `static/tutorial.json` with your audio clips and corresponding images.  
An example use case is already provided in the file, but here's a more formal explanation of the format:
```
{
  "0": ["static/tutorial/img0_1.jpg", ..., "static/tutorial/img0_12.jpg"],
  "1": ["static/tutorial/img1_1.jpg", ..., "static/tutorial/img1_12.jpg"],
  ...
  "N": ["static/tutorial/imgN_1.jpg", ..., "static/tutorial/imgN_12.jpg"]
}
```

Then update the following variables in `helpers.py`:
```python
number_of_tutorial_samples: int = N
INVITE_CODE = "supersecret"
BYPASS_REGISTRATION_KEY = "experiment006"
```

- `number_of_tutorial_samples` must match the number of entries in `tutorial.json`.  
- `INVITE_CODE` is required for user registration/login.  
- `BYPASS_REGISTRATION_KEY` allows one-time login via:  
  `https://your-domain.com/?key=<BYPASS_REGISTRATION_KEY>`

---

# How the Labeling Platform Works

1. Users register manually or via the `BYPASS_REGISTRATION_KEY`.  
2. On first login, users complete a short survey and tutorial.  
3. Labeling: users hear audio and select 1–4 matching images (this limit is configurable).  
4. At the end, users can optionally submit feedback.

![readme_figure](https://github.com/user-attachments/assets/71d91f82-3f88-4bb2-a6a3-5b2a9beec983)
