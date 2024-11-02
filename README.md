# 🏡 Linear Regression Web App with Streamlit

Welcome to the Linear Regression Web App! This app takes you on a journey through the famous California Housing dataset, allowing you to explore and model housing prices with ease—all powered by Streamlit! 🌟

## 🚀 What’s Streamlit?

[Streamlit](https://streamlit.io/) is a fantastic library that transforms Python scripts into interactive web applications. With just a few lines of code, you can create beautiful, data-driven applications, all without diving into complex web frameworks.

### Install Streamlit to get started!

```bash
pip install streamlit
```

# 🛠 Getting Started

To launch this app locally, we’ll use Docker to set up a containerized environment. 
You can get started in the web app in this repository by the following procedure. To prepare an environment, we use 
[docker](https://www.docker.com/).

1.  Create a docker image from Dockerfile.

    ```bash
    docker build -t (your_image_name) .
    ```

2.  Run a docker container created from the docker image. 

    ```bash
    docker run -it -p 8888:8888 -v ~/(local dir)/:/work (IMAGE ID) bash
    ```

    Note: The "-p 8888: 8888" connects the Docker container to your computer on port 8888, allowing you to access the app in your browser. The first and second 8888 indicate the host’s and the container's port numbers, respectively.

3.  Launch the App with Streamlit

    ```bash
    streamlit run regression_on_streamlit.py --server.port 8888
    ```

    Note: By specifying "--server.port 8888", you’ll be able to access the app in your browser at localhost:8888.

🌟 Enjoy Exploring the California Housing Dataset!