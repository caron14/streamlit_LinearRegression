FROM python:3.9.2

# EXPOSE 8888
WORKDIR /opt
RUN pip install --upgrade pip
RUN pip install streamlit==0.78.0 \
				numpy==1.20.1 \
				pandas==1.2.3 \
				matplotlib==3.3.4 \
				seaborn==0.11.1 \
				scikit-learn==0.24.1 

WORKDIR /work

# 
# docker run -it -p 8888:8888 -v ~/Desktop/code-practice/streamlit/code:/work <Image ID> bash
# streamlit run main.py --server.port 8888
# --> localhost:8888

