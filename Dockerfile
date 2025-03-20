FROM python:3.13.2

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app
RUN git clone https://github.com/abetlen/llama-cpp-python.git && \
    cd llama-cpp-python && \
    git submodule update --remote --merge && \
    pip install .

COPY --chown=user ./pyproject.toml pyproject.toml
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
CMD ["gradio", "ui.py", "--host", "0.0.0.0", "--port", "7860"]