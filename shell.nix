with import <nixpkgs> {};

let
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

in mkShell {
  nativeBuildInputs = [ qt5.qttools.dev ];

  propagatedBuildInputs = [
    (python3.withPackages (ps: with ps; [
      matplotlib
      pyqtwebengine
      pyqt5
      setuptools
      numpy
      scipy
      pandas
      pyopencl
      pyopengl
      pyopengl-accelerate 
      colorama
      xrt
      # my deps
      gitpython
      uncertainties
      jupyterlab
      plotly
    ]))

  ];

  # Normally set by the wrapper, but we can't use it in nix-shell (?).
  QT_QPA_PLATFORM_PLUGIN_PATH="${qt5.qtbase.bin}/lib/qt-${qt5.qtbase.version}/plugins";
  PYTHONPATH=builtins.getEnv "PWD"; 
  BASE_DIR=builtins.getEnv "PWD";
}

