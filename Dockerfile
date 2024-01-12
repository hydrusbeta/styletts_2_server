# Use Nvidia Cuda container base, sync the timezone to GMT, and install necessary package dependencies. Binaries are
# not available for some python packages, so pip must compile them locally. This is why gcc, g++, and python3.9-dev are
# included in the list below. Cuda 11.8 is used instead of 12 for backwards compatibility. Cuda 11.8 supports compute
# capability 3.5 through 9.0.
FROM nvidia/cuda:11.8.0-base-ubuntu20.04
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y --no-install-recommends \
    gcc \
    git \
    python3.9-dev \
    python3.9-venv \
    wget \
    espeak-ng

# Switch to a limited user
ARG LIMITED_USER=luna
RUN useradd --create-home --shell /bin/bash $LIMITED_USER
USER $LIMITED_USER

# Some Docker directives (such as COPY and WORKDIR) and linux command options (such as wget's directory-prefix option)
# do not expand the tilde (~) character to /home/<user>, so define a temporary variable to use instead.
ARG HOME_DIR=/home/$LIMITED_USER

# Download the pretrained LJSpeech model.
RUN mkdir -p ~/hay_say/temp_downloads/Models/LJSpeech/ && \
    wget https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth --directory-prefix=$HOME_DIR/hay_say/temp_downloads/Models/LJSpeech/ && \
    wget https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/config.yml --directory-prefix=$HOME_DIR/hay_say/temp_downloads/Models/LJSpeech/

# Create virtual environments for StyleTTS2 and Hay Say's styletts_2_server.
RUN python3.9 -m venv ~/hay_say/.venvs/styletts_2; \
    python3.9 -m venv ~/hay_say/.venvs/styletts_2_server

# Python virtual environments do not come with wheel, so we must install it. Upgrade pip while
# we're at it to handle modules that use PEP 517.
RUN ~/hay_say/.venvs/styletts_2/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel; \
    ~/hay_say/.venvs/styletts_2_server/bin/pip install --timeout=300 --no-cache-dir --upgrade pip wheel

# Install all python dependencies for StyleTTS2.
# Note: This is done *before* cloning the repository because the dependencies are likely to change less often than the
# StyleTTS2 code itself. Cloning the repo after installing the requirements helps the Docker cache optimize build time.
# See https://docs.docker.com/build/cache
RUN ~/hay_say/.venvs/styletts_2/bin/pip install \
    --timeout=300 \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    einops==0.7.0 \
    einops-exts==0.0.4 \
    librosa==0.10.1 \
    git+https://github.com/resemble-ai/monotonic_align.git@78b985be210a03d08bc3acc01c4df0442105366f \
    munch==4.0.0 \
    nltk==3.8.1 \
    phonemizer==3.2.1 \
    PyYAML==6.0.1 \
    torch==2.1.2+cu118 \
    torchaudio==2.1.2+cu118 \
    transformers==4.36.2

# install the 'punkt' tokenizer
RUN ~/hay_say/.venvs/styletts_2/bin/python -c "import nltk; nltk.download('punkt')"

# Install the dependencies for the StyleTTS2 interface code.
RUN ~/hay_say/.venvs/styletts_2_server/bin/pip install --timeout=300 --no-cache-dir \
    hay_say_common==1.0.8 \
    jsonschema==4.19.1

# Clone StyleTTS2 and checkout a specific commit that is known to work with this Docker file and with Hay Say.
RUN git clone -b main --single-branch -q https://github.com/yl4579/StyleTTS2 ~/hay_say/styletts_2
WORKDIR $HOME_DIR/hay_say/styletts_2
RUN git reset --hard 9b3dd4b910178088b1496a2f97d099f51c1058bb # Dec 16, 2023

# Clone the Hay Say interface code
RUN git clone -b main --single-branch -q https://github.com/hydrusbeta/styletts_2_server ~/hay_say/styletts_2_server

# Add command line functionality to StyleTTS2
RUN git clone -b main --single-branch -q https://github.com/hydrusbeta/styletts_2_command_line ~/hay_say/styletts_2_command_line && \
    mv ~/hay_say/styletts_2_command_line/command_line_interface.py ~/hay_say/styletts_2/

# Create directories that are used by the Hay Say interface code
RUN mkdir -p ~/hay_say/styletts_2/output/

# Expose port 6578, the port that Hay Say uses for RVC.
EXPOSE 6580

# Move the pretrained models to the expected directories.
RUN mv ~/hay_say/temp_downloads/Models ~/hay_say/styletts_2

# Execute the Hay Say interface code
CMD ["/bin/sh", "-c", "~/hay_say/.venvs/styletts_2_server/bin/python ~/hay_say/styletts_2_server/main.py --cache_implementation file"]