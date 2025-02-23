with import <nixpkgs> { };

let
  pythonPackages = python3Packages;
  xrt =
    let
      pname = "xrt";
      version = "1.6.0";
      extension = "zip";
    in
    python3Packages.buildPythonPackage {
      inherit pname version;
      src = fetchPypi {
        inherit pname version extension;
        sha256 = "1a2e19306abd67a4b45c8b9c4e05d7fb2d8a5836b82e08749d935bf4314599dc";
      };
      doCheck = false;
    };

in
pkgs.mkShell rec {
  name = "skif-xrt";
  venvDir = "./.venv";
  nativeBuildInputs = [ qt5.qttools.dev cmake ];

  buildInputs = [
    # adaptive deps
    pythonPackages.python
    pythonPackages.venvShellHook
    pythonPackages.numpy
    pythonPackages.scipy
    pythonPackages.pandas
    pythonPackages.ipykernel
    pythonPackages.ipywidgets
    pythonPackages.pyviz-comms
    pythonPackages.bokeh
    pythonPackages.mpi4py
    pythonPackages.cmake
    pythonPackages.scikit-build
    pythonPackages.selenium
    stdenv
    # xrt deps
    pythonPackages.matplotlib
    pythonPackages.pyqtwebengine
    pythonPackages.pyqt5
    pythonPackages.setuptools
    pythonPackages.pyopencl
    pythonPackages.pyopengl
    pythonPackages.pyopengl-accelerate
    pythonPackages.colorama
    xrt
    # my deps
    pythonPackages.gitpython
    pythonPackages.uncertainties
    pythonPackages.plotly

  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install --upgrade pip
    pip install jupyterlab
    pip install ipympl
    pip install "adaptive[notebook]"
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
  QT_QPA_PLATFORM_PLUGIN_PATH="${qt5.qtbase.bin}/lib/qt-${qt5.qtbase.version}/plugins";

  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  PYTHONPATH=builtins.getEnv "PWD"; 
  BASE_DIR=builtins.getEnv "PWD";

}
