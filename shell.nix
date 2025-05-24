with import <nixpkgs> { };
let 
  glebpkgs = fetchFromGitHub{
    owner = "glebdovzhenko";
    repo = "nixos-config";
    rev = "56289601413a1b2befccfd5d342a2dabaa4e7218";
    sha256 = "sha256-UrLRWqz3KgcALtMmeFmqmxZUoh0iG8u/s3bYZwHRP60=";
  };
in pkgs.mkShell rec {
  name = "skif-xrt";
  venvDir = "./.venv";
  nativeBuildInputs = [ qt5.qttools.dev cmake blas ];

  buildInputs = [
    # adaptive deps
    python3Packages.python
    python3Packages.venvShellHook
    python3Packages.numpy
    python3Packages.scipy
    python3Packages.pandas
    python3Packages.ipykernel
    python3Packages.ipywidgets
    python3Packages.pyviz-comms
    python3Packages.bokeh
    python3Packages.mpi4py
    python3Packages.cmake
    python3Packages.scikit-build
    python3Packages.selenium
    stdenv
    # xrt deps
    python3Packages.matplotlib
    python3Packages.pyqtwebengine
    #python3Packages.pyqt5-webkit
    python3Packages.pyqt5
    python3Packages.setuptools
    python3Packages.pyopencl
    python3Packages.pyopengl
    python3Packages.pyopengl-accelerate
    python3Packages.colorama
    (callPackage "${glebpkgs}/pkgs/xrt" { })

    # my deps
    python3Packages.gitpython
    python3Packages.uncertainties
    python3Packages.plotly
    python3Packages.dash
  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install --upgrade pip
    pip install jupyterlab
    pip install ipympl
    pip install "adaptive[notebook]"
    pip install siphash24
    pip install pymc
    pip install bambi
    pip install arviz
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    jupyter labextension install @pyviz/jupyterlab_pyviz
    python -m ipykernel install --user --name=${name}
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
  '';
  QT_QPA_PLATFORM_PLUGIN_PATH = "${qt5.qtbase.bin}/lib/qt-${qt5.qtbase.version}/plugins";

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  PYTHONPATH = builtins.getEnv "PWD";
  BASE_DIR = builtins.getEnv "PWD";

}
