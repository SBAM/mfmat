from conans import ConanFile, CMake, tools
from conans.errors import ConanInvalidConfiguration
import os

class MfmatConan(ConanFile):
  name = "mfmat"
  homepage = "https://github.com/SBAM/mfmat"
  url = "https://github.com/SBAM/mfmat"
  description = "MF linear algebra library"
  settings = "os", "compiler", "build_type", "arch", "cppstd"
  options = { "shared": [True, False],
              "fPIC": [True, False],
              "toolchain_file": "ANY",
              "lto": [True, False] }
  no_copy_source = True
  generators = "cmake", "cmake_find_package"
  requires = ( "boost/1.81.0",
               "cmake/3.25.1",
               "lz4/1.9.4",
               "pybind11/2.10.1" )
  default_options = { "shared": False,
                      "fPIC": True,
                      "lto": False,
                      "boost:system_no_deprecated": True,
                      "boost:asio_no_deprecated": True,
                      "boost:filesystem_no_deprecated": True,
                      "cmake:with_openssl": False }

  # def system_requirements(self):
  #   if os_info.linux_distro == "fedora":
  #     Dnf(self, "noarch").install(["opencl-headers"])
  #     Dnf(self).install(["rocm-opencl-devel"])

  def set_version(self):
    git = tools.Git()
    git.check_repo()
    if os.environ.get("GIT_TAG") is None:
      self.output.info("Deducing GIT_TAG from git repository")
      self.version = git.run("describe --tags --always")
    else:
      self.output.info("Using environment GIT_TAG")
      self.version = os.environ.get("GIT_TAG")

  def config_options(self):
    git = tools.Git()
    self.output.info("GIT_TAG={}".format(self.version))
    tmp_an = git.run("log -1 --format='%aN'")
    tmp_ae = git.run("log -1 --format='%aE'")
    self.git_author = "{} <{}>".format(tmp_an, tmp_ae)
    self.output.info("GIT_AUTHOR={}".format(self.git_author))
    self.git_commit_date = git.run("log -1 --format='%ci'")
    self.output.info("GIT_COMMIT_DATE={}".format(self.git_commit_date))
    self.git_commit_hash = git.get_commit()
    self.output.info("GIT_COMMIT_HASH={}".format(self.git_commit_hash))

  def validate(self):
    if not self.options.toolchain_file:
      raise ConanInvalidConfiguration("Toolchain file unspecified")

  def configure_cmake(self):
    cmake = CMake(self)
    cmake.definitions["CMAKE_TOOLCHAIN_FILE"] = self.options.toolchain_file
    cmake.definitions["SHARED"] = "ON" if self.options.shared else "OFF"
    cmake.definitions["LTO"] = "ON" if self.options.lto else "OFF"
    cmake.definitions["GIT_AUTHOR"] = self.git_author
    cmake.definitions["GIT_COMMIT_DATE"] = self.git_commit_date
    cmake.definitions["GIT_COMMIT_HASH"] = self.git_commit_hash
    cmake.definitions["GIT_TAG"] = self.version
    cmake.configure(args=["--no-warn-unused-cli"])
    return cmake

  def build(self):
    cmake = self.configure_cmake()
    cmake.parallel = True
    cmake.build()

  def test(self):
    cmake = self.configure_cmake()
    cmake.parallel = False
    cmake.test()
