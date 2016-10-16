FROM fedora:23
 
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN localedef --quiet -c -i en_US -f UTF-8 en_US.UTF-8
 
RUN dnf -y update && dnf -y install \
    bzip2 \
    tar \
    wget \
    git \
    && dnf clean all
 
# Configure environment
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH
 
# Install conda
RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh && \
    echo "6c6b44acdd0bc4229377ee10d52c8ac6160c336d9cdd669db7371aa9344e1ac3 *Miniconda3-3.9.1-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-3.9.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-3.9.1-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda install --yes conda==3.14.1
 
RUN conda install --yes libgfortran libgcc
RUN pip install spacy && python -m spacy.en.download -f
RUN conda install --yes joblib numpy matplotlib pandas scikit-learn

COPY scripts/ /tmp/scripts/
COPY ext/ /tmp/ext/
COPY execute_all.sh /tmp/scripts
WORKDIR /tmp/scripts
ENTRYPOINT ["/bin/sh", "/tmp/scripts/execute_all.sh"]