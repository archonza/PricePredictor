FROM library/python:3.7-bookworm

ARG USER=vscode
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y --no-install-recommends sudo \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /usr/bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER} \
    && pip3 install pandas \
    && pip3 install scikit-learn \
    && pip3 install matplotlib \
    && git config --global user.email "rudolferasmus@gmail.com" \
    && git config --global user.name "rudolferasmus"
