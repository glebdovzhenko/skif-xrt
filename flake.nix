{
  description = "My xrt beamline modeling flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    glebpkgs.url = "github:glebdovzhenko/nixos-config";

  };

  outputs =
    { self
    , nixpkgs
    , flake-utils
    , glebpkgs
    , ...
    }:
    let
      overlays = [ ];
      systems = [ "x86_64-linux" ];
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = import nixpkgs { inherit overlays system; };
        xrt = pkgs.callPackage "${glebpkgs}/pkgs/xrt" { };
      in
      {
        devShells.default = pkgs.mkShell
          rec {
            name = "skif-xrt";
            venvDir = "./.venv";
            nativeBuildInputs = with pkgs; [ qt5.qttools.dev cmake blas ];

            buildInputs = with pkgs; [
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
              python3Packages.pyqt5
              python3Packages.setuptools
              python3Packages.pyopencl
              python3Packages.pyopengl
              python3Packages.pyopengl-accelerate
              python3Packages.colorama
              # my deps
              python3Packages.gitpython
              python3Packages.uncertainties
              python3Packages.plotly
              python3Packages.dash
            ] ++ [ xrt ];

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
              # setting up project env
              export BASE_DIR=$PWD;
              export PYTHONPATH=$PYTHONPATH":"$PWD;
            '';
            QT_QPA_PLATFORM_PLUGIN_PATH = "${pkgs.qt5.qtbase.bin}/lib/qt-${pkgs.qt5.qtbase.version}/plugins";
            LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
          };
        devShell = self.devShells.${system}.default;
      }
    );
}
