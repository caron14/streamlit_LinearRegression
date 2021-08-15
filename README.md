# Web app. of Linear regression by Streamlit

Implementation of linear regression of the Boston house prices dataset on a web app by Streamlit.

## About library for deploying a web app

We create a web app by 
[Streamlit](https://streamlit.io/),
a python library for turning a python script into a web app.

You can easily install and get started in streamlit.

```bash
pip install streamlit
```

## Getting started

You can get started in the web app in this repository by the following procedure. To prepare an environment, we use 
[docker](https://www.docker.com/).

1.  Create a docker image from Dockerfile.

    ```bash
    docker build .
    ```

2.  Run a docker container created from the docker image. 

    ```bash
    docker run -it -p 8888:8888 -v ~/(local dir)/:/work (IMAGE ID) bash
    ```

    Note that "-p 8888: 8888" is an instruction to connect the host(your local PC) with the docker container. The first and second 8888 indicate the host’s and the container's port numbers, respecticely.

3.  Turn a python script into a web app by Streamlit

    ```bash
    streamlit run regression_on_streamlit.py --server.port 8888
    ```

    Note that, by "--server.port 8888", we can access a web app from a web browser with the URL "localhost: 8888".