FROM ubuntu:20.04 as dolfinsrc
# Special image compiled with OpenMPI for compliance with Sigma2 HPC.
# Adapted from Jørgen Dokken, https://fenicsproject.discourse.group/t/how-to-install-dolfin-on-ubuntn-20-04-from-source/4233/4

ARG PYBIND11_VERSION=2.5.0
ARG PETSC_VERSION=3.13.3
ARG SLEPC_VERSION=3.13.3
ARG PETSC4PY_VERSION=3.13.0
ARG SLEPC4PY_VERSION=3.13.0

# Flags to pass to ninja build system.
ARG MAKEFLAGS
# Compiler optimisation flags for SLEPc and PETSc, all languages.
ARG PETSC_SLEPC_OPTFLAGS="-O2 -march=sandybridge"
# Turn on PETSc and SLEPc debugging. "yes" or "no".
ARG PETSC_SLEPC_DEBUGGING="no"
# CMake build type for DOLFIN C++ build. See CMake documentation.
ARG DOLFIN_CMAKE_BUILD_TYPE="RelWithDebInfo"
# Extra CMake C++ compiler flags for DOLFIN C++ build.
ARG DOLFIN_CMAKE_CXX_FLAGS="-march=sandybridge"
# Ubuntu MPI variant. "mpich" or "openmpi".
ARG MPI="openmpi"

WORKDIR /tmp

# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0

# Install dependencies available via apt-get.
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or tests.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    clang-10 \
    cmake \
    g++ \
    gfortran \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libboost-math-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-timer-dev \
    libeigen3-dev \
    libhdf5-${MPI}-dev \
    liblapack-dev \
    lib${MPI}-dev \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools \
	wget

# Install Python packages (via pip)
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
RUN pip3 install --no-cache-dir mpi4py numba && \
    pip3 install --no-cache-dir cffi cppimport flake8 pytest pytest-xdist sphinx sphinx_rtd_theme

# Install pybind11
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make install && \
    rm -rf /tmp/*


# Install PETSc with real and complex types
ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc
WORKDIR /tmp
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p ${PETSC_DIR} && tar -xf petsc-${PETSC_VERSION}.tar.gz -C ${PETSC_DIR} --strip-components 1 && \
    cd ${PETSC_DIR} && \
    # Real, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-real-32 \
    --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-blacs \
    --download-hypre \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-spai \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=real && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-real-32 ${MAKEFLAGS} all && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install SLEPc
WORKDIR /tmp
RUN wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p ${SLEPC_DIR} && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C ${SLEPC_DIR} --strip-components 1 && \
    cd ${SLEPC_DIR} && \
    export PETSC_ARCH=linux-gnu-real-32 && \
    python3 ./configure && \
    make && \
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*

# Install petsc4py and slepc4py with real and complex types
RUN PETSC_ARCH=linux-gnu-real-32 pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION} slepc4py==${SLEPC4PY_VERSION}

WORKDIR /src
RUN apt-get -qq update && \
    apt-get -y install git
# This leaves the sources inside the container. This is a limitation of Docker.
# There is some trickery in the intermediate and dolfin containers that can be
# used to remove this source if needed, see below.
RUN git clone https://github.com/FEniCS/fiat.git && \
    git clone https://bitbucket.org/fenics-project/ffc.git && \
    git clone --branch 2022.1.0 https://github.com/FEniCS/ufl.git && \
    git clone https://github.com/FEniCS/ufl-legacy.git && \
    git clone https://bitbucket.org/fenics-project/dolfin.git && \
    git clone https://bitbucket.org/fenics-project/dijitso.git

# The dolfin-onbuild container expects to have folders fiat/ ufl/ ffc/ and
# dolfin/ mounted/shared at /src.
RUN cd fiat && pip3 install --no-cache-dir . && \
    cd ../ufl && pip3 install --no-cache-dir . && \
    cd ../ufl-legacy && pip3 install --no-cache-dir . && \
    cd ../dijitso && pip3 install --no-cache-dir . && \
    cd ../ffc && pip3 install --no-cache-dir . && \
    cd ../ && pip3 install --no-cache-dir ipython

RUN cd dolfin && \
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-real-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfin -DCMAKE_BUILD_TYPE=${DOLFIN_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFIN_CMAKE_CXX_FLAGS} .. && \
    ninja ${MAKEFLAGS} install && \
    cd ../python && \
    PETSC_ARCH=linux-gnu-real-32 pip3 install --target /usr/local/dolfin/lib/python3.8/dist-packages --no-dependencies .


# Real by default.
ENV PKG_CONFIG_PATH=/usr/local/dolfin/lib/pkgconfig:$PKG_CONFIG_PATH \
            PETSC_ARCH=linux-gnu-real-32 \
            PYTHONPATH=/usr/local/dolfin/lib/python3.8/dist-packages:$PYTHONPATH \
            LD_LIBRARY_PATH=/usr/local/dolfin/lib:$LD_LIBRARY_PATH

WORKDIR /src
RUN apt-get -qq update && \
    apt-get -y install pkg-config && \
	pip3 install pkgconfig
    
RUN git clone https://github.com/gautelinga/twoasis.git
RUN cd twoasis && pip3 install --no-cache-dir .

WORKDIR /root

# Note that because we inherit from dev-env we do not inherit these ENV from
# dolfin-onbuild.
ENV PKG_CONFIG_PATH=/usr/local/dolfin/lib/pkgconfig:$PKG_CONFIG_PATH \
    PETSC_ARCH=linux-gnu-real-32 \
    PYTHONPATH=/usr/local/dolfin/lib/python3.8/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfin/lib:$LD_LIBRARY_PATH
